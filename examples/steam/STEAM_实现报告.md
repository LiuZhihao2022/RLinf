# STEAM 代码实现报告（RLinf）

> 适用仓库：`/mnt/public/guqiuyi/RLinf_active`
> 论文：`STEAM: Self-Supervised Temporal Ensemble Advantage Modeling for Real-World Robot Learning`（CoRL 2026 投稿，`STEAM_arXiv_v1.pdf`）
> 本报告覆盖：算法↔代码的对应、三大步骤（训练 advantage 模型 / 计算 advantage / CFG 训练）的实现细节、以及对运行流程的核对。

---

## 0. 一句话总览

STEAM 用**专家示范的时间结构**做自监督，训练一个**温度偏移（temporal-offset）分类器的集成**，把每个 `(frame_t, frame_{t+k})` 帧对映射成一个 `[-1, 1]` 的**有符号 advantage**；集成用 **worst-of-N（逐样本取 min）** 抑制 OOD 上的过高估计；最后把 advantage 量化成 0/1 最优性标签 `o`，通过 **CFGRL** 引导 π0/OpenPI flow-matching 策略。

代码落在三个目录：

| 步骤 | 目录 | 入口脚本 | 训练/计算的 Worker 或脚本 |
|---|---|---|---|
| Step 1 训练 advantage（value）模型 | `examples/steam/value/` | `train_steam.py` | `rlinf/workers/sft/fsdp_steam_sft_worker.py` |
| Step 2 计算 advantage（离线打标） | `examples/steam/process/` | `compute_advantages_ensemble.py` | 同文件（torchrun 多卡） |
| Step 3 CFG 策略训练 | `examples/steam/cfg/` | `train_cfg.py` | `rlinf/workers/sft/fsdp_cfg_worker.py` |

三步都由同一个 `SFTRunner`（`rlinf/runners/sft_runner.py`）驱动训练循环。

---

## 1. 算法核心 → 代码映射

### 1.1 自监督目标：归一化时间偏移（论文 §3.1, Eq.1–2）

对专家 episode `τ_k`，帧对 `(f_{k,i}, f_{k,j})` 的偏移 `Δ = j − i`，再按轨迹长度归一化 `Δ̃ = (j−i)·L_max/L_τk`。
- **正样本**：`(t, t+k)` 正序 → progress；**负样本**：把专家轨迹**倒放** `(t+k, t)` → regress（无需失败示范即可学到"退步"）。
- 代码：`rlinf/data/datasets/steam/pair_dataset.py`
  - `PairDataset.__getitem__` 对每个锚点 `t` 同时产出正/负两个样本（`pair_dataset.py:881-928`）。
  - 长度归一化由 `length_scale_enabled` / `length_scale_percentile`（默认 90 分位作 `L_max`）控制；`set_length_scale_reference()` 计算 per-dataset 的 `L_max`，混合数据集可用 `compute_global_length_scale_reference()` 统一。

### 1.2 分布式偏移预测 + 分箱（论文 §3.2, Eq.3）

把连续偏移 `Δ̃` 离散成 `N` 个**有符号 bin**，做 `N` 路交叉熵（带 label smoothing）。
- 分箱数学：`rlinf/data/datasets/steam/binning.py`
  - `_signed_stride_to_bin(stride, K, num_bins)`：把有符号 stride `∈ {-K..-1, 1..K}` 映射到 bin idx。`[0, N/2)` 为 regress（负），`[N/2, N)` 为 progress（正）。
  - 约束：`num_bins` 为偶数且 `2K % num_bins == 0`（构造时校验）。
  - `_scaled_signed_stride_to_bin`：长度归一化版（短 episode 更早饱和到 `±K`）。
  - `bin_centers` / `expected_signed_stride`：bin 概率 → 期望有符号 stride。
- 模型 backbone：`rlinf/models/embodiment/steam/modeling_steam.py::SteamBackbone`
  - **SigLIP（vision）+ Gemma-3-270M（language）+ MLP 融合头**。
  - 每帧分别过 vision encoder，**逐相机 mean-pool（带 mask）**，得到 per-frame 特征 `[B, num_frames, D]`；语言特征 mean-pool over tokens。
  - 关键：**per-frame 特征按顺序 concat**（不是跨帧 pool），保留 `(t, t+k)` 的方向信息 → `_fuse()` 输出 `[B, D·(num_frames+1)]` → `value_head` 输出 `[B, num_bins]`。
- Critic 封装：`rlinf/models/embodiment/steam/modeling_critic.py::SteamCriticModel`
  - `forward(observation, labels)` → `CriticOutput`；`_compute_loss` 为 `num_bins` 路交叉熵（`label_smoothing=cfg.label_smoothing`）。
  - **advantage 标量**：`_predicted_signed_value(probs)` = `Σ_b p_b · signed_bin_b / (N/2)`，范围 `[-1, 1]`。binary（`num_bins=2`）退化为 `2·P(progress) − 1`。这正是论文 Eq.4 的"期望 bin − 真值偏移"在实现中的有符号期望形式。

### 1.3 集成抑制过高估计（论文 §3.2, Eq.5；Algorithm 1）

`A_STEAM = min_m A(f; θ_m)`（worst-of-N）。
- `rlinf/models/embodiment/steam/ensemble_modeling_critic.py::EnsembleSteamCriticModel`
  - `predict()`：对每个 member 求 `predicted_values`，`_aggregate_member_predictions` 用 `member_predicted_values.min(dim=0)` 取**逐样本最小**作为聚合 advantage（`prediction_min`），同时输出 `mean/variance/member_*`（仅诊断）。
  - **训练**：`forward(..., member_idx=m)` 只跑第 `m` 个 member，由 worker 外层循环逐 member 喂**各自独立的随机 micro-batch**（bagging 风格，使集成方差成为有意义的认知不确定性信号）。
  - member 构造：`clone_ensemble_members` 复制 backbone，`reinitialize_member_value_heads(head_seed_base + idx)` 给每个 head 不同随机种子。

### 1.4 量化为最优性标签 + CFGRL（论文 §3.3, Eq.6–11；Algorithm 2）

对每个数据源**分别**做分位数阈值：`o = 1[A_STEAM ≥ δ_q]`，其中 expert 用 `ϖ_exp`、non-expert 用 `ϖ_non-exp`。
- 打标：`examples/steam/process/compute_advantages_ensemble.py`（见 §3）。
- CFGRL 训练：`rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py`（见 §4）。flow-matching：`x_t=(1−t)a + tε`，目标速度 `u_t = ε − a`，MSE 损失；按 `unconditional_prob` 随机 drop 条件做 classifier-free guidance。

---

## 2. Step 1：训练 advantage（value）模型

**入口**：`examples/steam/value/train_steam.py` → `FSDPSteamSftWorker` + `SFTRunner`。
**启动**：`bash examples/steam/value/run_steam_sft.sh steam_model_ensemble1`
**配置**：`examples/steam/value/config/steam_model_ensemble1.yaml`（+ `config/model/steam.yaml` + 共享 `examples/sft/config/training_backend/fsdp.yaml`）。

### 2.1 模型构造
- `model_provider_func`（`fsdp_steam_sft_worker.py:196-205`）：
  - `_ensure_steam_precision_cfg`：若未设 `precision` 则默认 `fp32`（FSDP master 用 fp32 防 Adam 二阶矩塌缩；前向 dtype 由 `fsdp_config.mixed_precision.param_dtype` 单独控制，本配置是 bf16）。
  - 读 `cfg.actor.model.ensemble_size`；当 `>1` 且 `ensemble_head_seed_base` 为 None 时，回退用 `cfg.actor.seed` 作种子基。
  - `get_model(cfg.actor.model)`（`steam/__init__.py`）：`ensemble_size==1` → `SteamCriticModel`；`>1` → `EnsembleSteamCriticModel`（克隆 backbone + reinit heads）。

### 2.2 数据与 collator
- 数据集：`PairDataset`（单个）/ `PairMixtureDataset`（多个加权混合）。
- `data.k` 写入 `model_cfg.stride_k`；当前配置 `k=32`、`num_bins=32`（满足 `2·32 % 32 == 0`）。
- collator：`BinaryPairDataCollator`，对 `frame_t`、`frame_tk` 各跑一次 `SteamProcessor`，再沿 frame 轴 stack；输出 `observation`（多相机 `[B, 2, 3, H, W]` + mask + tokenized prompt）和 `labels`（`long`，bin idx）。
- `only_success: true`（专家成功示范）。相机：`[face_view, left_wrist_view, right_wrist_view]`。

### 2.3 训练步与集成
- `_backward_one_micro_batch`：`model(observation, labels, member_idx=…)`；loss 直接取自 `CriticOutput.loss`，按 `1/grad_accum` 缩放。
- 集成训练 `_run_training_members`：**顺序**遍历每个 member，各跑 `grad_accum` 个 micro-batch 后各做一次 optimizer step；每个全局 step 消耗 `ensemble_size × grad_accum` 个 batch。峰值显存≈单 member（顺序执行）。
- `get_max_steps_per_epoch = len(loader) // (grad_accum × ensemble_size)`。

### 2.4 checkpoint 输出布局
`SFTRunner._save_checkpoint`（每 `save_interval` 步）→ `actor.save_checkpoint`：

```
{runner.logger.log_path}/{experiment_name}/checkpoints/
    global_step_{N}/
        actor/
            model_state_dict/full_weights.pt   # FSDP 合并权重
            config.json                        # SteamConfig（含 ensemble_size/num_bins/stride_k/…）
            tokenizer.json / tokenizer_config.json / preprocessor_config.json …
    best_model/actor/                          # 若启用 early stop
```
- `config.json` 由 `save_steam_checkpoint_assets` 写出 → Step 2 的 `SteamCriticModel.from_checkpoint` 据此重建模型（**读 `config.json` 的 `ensemble_size` 决定单/集成**）。

### 2.5 关键超参（当前 ensemble1.yaml）
`max_steps=16000`、`save_interval=4000`、`micro_batch_size=16`、`global_batch_size=512`、`lr=5e-5`、`vision=siglip2-so400m-patch14-224`、`language=gemma-3-270m`、`ensemble_size=1`、`data.seed=7`、`actor.seed=0`、`num_nodes=2`。

> 对照论文 Table 4：`N=32`、`max offset=32`、`M=3`、`H=32`、`lr=5e-5`、`batch=512`（towel/chip/cola）。当前 `num_bins=32`、`k=32` ✅；但 `ensemble_size=1`，要凑 `M=3` 需训练 3 个并合并（见 §6 流程核对）。

---

## 3. Step 2：计算 advantage（离线打标）

**入口/脚本**：`examples/steam/process/compute_advantages_ensemble.py`（支持 `torchrun` 多卡）。
**启动**：`bash examples/steam/process/run_compute_advantages_ensemble.sh`（自动按 GPU 数起 torchrun）。
**配置**：`examples/steam/process/config/compute_advantages_ensemble.yaml`。

### 3.1 流程
1. `setup_distributed` → 各 rank 分片（`get_shard_indices`）。
2. `SteamCriticModel.from_checkpoint(value_checkpoint, …)` 加载模型；`_coerce_inference_model` 把 `ensemble_size=1` 的单模型也包装成 `EnsembleSteamCriticModel`（M=1）以统一接口。
3. **Phase 1**：对每个数据集用 `BinaryPairInferenceDataset`（前向锚点 `t∈[0,T-2]`，每个锚点 `(t, t+k)`）跑 `model.predict`，得到每帧 `ensemble_signed_score = predicted_values`（= worst-of-N 的 `prediction_min`）以及 `mean/min/variance/member_values/entropy/expected_stride_normalized`。rank0 聚合（`gather_dataframes_to_rank0`），并对每个 episode 末帧补 0 分。`advantage_continuous = ensemble_signed_score`。
4. **Phase 2（仅 rank0）**：按 `label_mode` 决定布尔 `advantage`：
   - `threshold`：`advantage = advantage_continuous > positive_threshold`（阈值在 `[-1,1]`，rollout 才比较；sft 全 True）。
   - `quantile`：rollout 池取 top `rollout_quantile`，sft 池（若设 `expert_quantile`）取 top `expert_quantile`，**两池独立**算分位阈值。
5. 写出 `meta/advantages_{tag}.parquet`（列：`episode_index, frame_index, advantage, advantage_continuous, ensemble_signed_score, p_progress_mean/min/variance, member_values, expected_stride_normalized, entropy_*`），并更新 `meta/mixture_config.yaml` 的 `tags[tag]`。

### 3.2 当前配置 vs 论文
- `label_mode: quantile`、`rollout_quantile: 0.3`、`expert_quantile: 0.8` → **与论文 Table 4 完全一致**（`ϖ_non-exp=0.3`、`ϖ_exp=0.8`）。
- `data.k: 32` 必须与训练时 `data.k` 一致（否则时间尺度错配，静默劣化）。✅ 与 Step 1 一致。
- `camera_keys` 必须与训练一致。✅。
- `tag: steam_k32_ensemble3_ckpt16000_wco_exp_and_rollout` → **Step 3 的 `data.advantage_tag` 必须用同一个 tag**。✅（两边当前一致）。
- ⚠️ `value_checkpoint`、`train_data_paths`（sft/rollout）目前是 `/path/...` 占位，需填真实路径；且应指向**合并后的 ensemble checkpoint**（见 §6）。

### 3.3 辅助脚本
- `merge_steam_ensemble.py`：**把多个单 seed checkpoint 合并成一个 ensemble checkpoint**（关键，见 §6）。`--member PATH[:idx]`（可多次）`--output OUT/actor`，写出 `config.json`（`ensemble_size=len(members)`）、`model_state_dict/full_weights.pt`（key 为 `members.i.model.*`）、`merge_manifest.json` + 拷贝 tokenizer/processor。会校验各 member 架构兼容（`num_bins/stride_k/fusion_hidden_dim/...`）。
- `relabel_advantages.py`：**纯 CPU**，从已有 `advantages_{source_tag}.parquet` 的 `advantage_continuous` 重新派生布尔标签到 `{new_tag}`（换阈值/分位时不必重跑 GPU）。
- `visualize_advantage.py`：读 parquet 出 5 张图（分布/per-member/不确定性散点/per-episode 正例率/episode 时间线）+ `summary.json`。对应论文 Fig.4–6、Fig.11 的可视化。

---

## 4. Step 3：CFG 策略训练

**入口**：`examples/steam/cfg/train_cfg.py` → `FSDPCfgWorker` + `SFTRunner`。
**启动**：`bash examples/steam/cfg/run_cfg_sft.sh x2robot_cfg_openpi`
**配置**：`examples/steam/cfg/config/x2robot_cfg_openpi.yaml`。

### 4.1 数据路径
- `FSDPCfgWorker.build_dataloader`：对每个 `train_data_paths` 项，加载 LeRobot 数据 + `_load_advantages_lookup` 读 `meta/advantages_{advantage_tag}.parquet` → `(episode_index, frame_index) → advantage(bool)`，包成 `AdvantagePreservingDataset`（在样本里注入 `advantage`），再 `CfgMixtureDataset` 按 `weight` 加权（`balance_dataset_weights` 时按数据集长度再平衡）。
- `CFGDataLoaderImpl` 产出 `(observation, actions, advantage)`。

### 4.2 模型与 CFGRL
- 模型：`OpenPi0ForCFGActionPrediction`（包 π0/OpenPI，`model_type: cfg_model`）。
- 路由 `compute_cfg_routing_masks`：用 `advantage`（bool）+ `unconditional_prob` + `positive_only_conditional` 决定每个样本走 conditional/unconditional 分支。
  - `positive_only_conditional=True`（当前配置）：只有正样本可进 conditional（按 `unconditional_prob` 随机 drop），负样本恒 unconditional。
- 语言条件 `TokenizePromptWithGuidance`：基础 prompt / `"{prompt}\nAdvantage: positive"` / `"...negative"` 三套 token。
- flow-matching 损失 `_compute_flow_losses`：`t~U[0,1)`、`ε~N(0,I)`、`x_t = tε+(1−t)a`、目标 `u_t = ε−a`、`MSE(v_θ, u_t)`。**对应论文 Eq.8–9**。
- 推理 `sample_actions`：Euler 积分，`v = (1−w')·v_uncond + w'·v_cond`（`w' = cfgrl_guidance_scale`）。**对应论文 Eq.10–11**。

### 4.3 当前配置要点
`model_path=pi0_base_pytorch`、`model_type=cfg_model`、`action_dim=28`、`num_action_chunks=20`、`openpi.config_name=restock_cola_sm2sm`（已在 `dataconfig/__init__.py:377` 注册）、`asset_id=restock_cola_recap`、`unconditional_prob=0.1`、`positive_only_conditional=true`、`cfgrl_guidance_scale=1.0`（训练期；推理引导强度论文用 `w=2.5`）、`lr=5e-5`、`total_training_steps=30000`、`advantage_tag` 与 Step 2 一致 ✅。

> 对照论文 Table 4：`p_drop=0.1` ✅、`w=2.5`（推理）、`lr=5e-5` ✅、`steps=30000` ✅、`batch=512`（注意当前 `global_batch_size=256`，与论文 512 不同，可按需调）。

---

## 5. 端到端数据流（一图流）

```
专家 sft 数据 ──(PairDataset 正/倒序帧对)──▶ [Step1] 训练 M 个 temporal-offset 预测器
                                                     │ 各自 checkpoint (ensemble_size=1)
                                                     ▼
                                   merge_steam_ensemble.py 合并 ──▶ ensemble ckpt (ensemble_size=M, config.json)
                                                     │
专家 sft + rollout 数据 ──(BinaryPairInferenceDataset)──▶ [Step2] compute_advantages_ensemble
                                                     │  worst-of-N min 聚合 → advantage_continuous
                                                     │  分位阈值(0.8 exp / 0.3 rollout) → advantage(bool)
                                                     ▼
                              每个数据集 meta/advantages_{tag}.parquet  +  mixture_config.yaml[tags][tag]
                                                     │  (advantage_tag 对齐)
                                                     ▼
专家 sft + rollout 数据 ──(AdvantagePreservingDataset 注入 o)──▶ [Step3] CFGRL 训练 π0
                                                     ▼
                                              引导后的 VLA 策略
```

---

## 6. 运行流程核对（对你写的 step0–step3 的确认）

整体**步骤顺序正确、配置文件对应正确**，但有 **1 个必补步骤** 和 **几个易错点**：

### 🔴 必补：Step 1 与 Step 2 之间缺「合并 ensemble」一步

- 论文默认 `M=3`（Table 3 显示 M=1 仅 72.7%，M=3 达 92.3%）；STEAM 的核心 worst-of-N min 聚合**只有在多 member 时才生效**。
- `steam_model_ensemble1.yaml` 是 `ensemble_size: 1`（单预测器）。你按注释"改下 seed 跑多个"会得到**多个独立的单 member checkpoint**，但 Step 2 只加载**一个** checkpoint，并按其 `config.json` 的 `ensemble_size` 决定集成大小。
- 因此必须在 Step 1 之后、Step 2 之前，用 `merge_steam_ensemble.py` 把 3 个单 seed checkpoint 合并成一个 `ensemble_size=3` 的 checkpoint，再让 Step 2 的 `value_checkpoint` 指向合并产物。
- 若跳过合并、直接指向单个 checkpoint：能跑通，但等价于 **M=1**（无 min 抑制），且与 tag 名 `ensemble3` 不符。

补充后的流程（推荐）：

```bash
# Step 1：训练 3 个独立预测器（每个改 seed + 改 experiment_name，避免互相覆盖）
bash examples/steam/value/run_steam_sft.sh steam_model_ensemble1 \
    data.seed=7  runner.logger.experiment_name=steam_sft_<task>_seed7
bash examples/steam/value/run_steam_sft.sh steam_model_ensemble1 \
    data.seed=42 runner.logger.experiment_name=steam_sft_<task>_seed42
bash examples/steam/value/run_steam_sft.sh steam_model_ensemble1 \
    data.seed=17 runner.logger.experiment_name=steam_sft_<task>_seed17

# Step 1.5（必补）：合并成 ensemble_size=3 的 checkpoint
python examples/steam/process/merge_steam_ensemble.py \
    --member <log_path>/steam_sft_<task>_seed7/checkpoints/global_step_16000/actor \
    --member <log_path>/steam_sft_<task>_seed42/checkpoints/global_step_16000/actor \
    --member <log_path>/steam_sft_<task>_seed17/checkpoints/global_step_16000/actor \
    --output  <某目录>/steam_value_ensemble3/checkpoints/global_step_16000/actor

# Step 2：value_checkpoint 指向合并后的 actor 目录
bash examples/steam/process/run_compute_advantages_ensemble.sh \
    advantage.value_checkpoint=<某目录>/steam_value_ensemble3/checkpoints/global_step_16000/actor

# Step 3：照旧
bash examples/steam/cfg/run_cfg_sft.sh x2robot_cfg_openpi
```

> 备选方案：直接把 `steam_model_ensemble1.yaml` 的 `ensemble_size` 改成 3 跑**一次**训练（worker 支持顺序多 member 训练），就**不需要 merge**。但这样各 member 共享同一份 backbone 克隆、仅 head 不同 seed + 数据 bagging；而"分开跑 3 个 seed + 合并"得到的是 backbone 与数据都完全独立的 member，更贴近论文 Algorithm 1。当前命名 `ensemble1` + 注释明显是走"分开跑 + 合并"路线。

### 🟡 易错点清单

1. **改 seed 的同时必须改 `experiment_name`**。当前 `experiment_name` 固定为 `steam_sft_clean_table_ygg06170618`，3 个 seed 用同名会写到**同一 checkpoint 目录互相覆盖**。务必每个 seed 给不同 `experiment_name`（如上）。需要更强独立性时可同时改 `actor.seed`。

2. **三步要对准「同一个任务」的数据集**。当前模板里：Step1 训练用 `clean_table_ygg06170618_steam_sft`，而 Step2/Step3 用 `restock_cola_*`——这是不同任务的占位示例。实际跑一个任务时，三步要一致：
   - Step1：该任务的**专家 sft**（训练预测器）；
   - Step2：该任务的**专家 sft + rollout**（打标，parquet 写进各自 `meta/`）；
   - Step3：与 Step2 **相同的数据集列表**（按 `advantage_tag` 读 parquet）。

3. **`advantage_tag` 必须三处对齐**：Step2 写出的 `advantage.tag` ＝ Step3 `data.advantage_tag`。当前两边都是 `steam_k32_ensemble3_ckpt16000_wco_exp_and_rollout` ✅（建议 tag 里的 `ckpt16000` 与实际合并所用的 `global_step` 对上，纯命名规范）。

4. **`data.k` / `num_bins` / `camera_keys` 必须 Step1 与 Step2 一致**。当前 `k=32`、`camera_keys` 三视角一致 ✅。`num_bins`（Step1=32）由 checkpoint `config.json` 带入 Step2，无需在 Step2 重设。

5. **Step2 的占位路径要填真实值**：`advantage.value_checkpoint`、`data.train_data_paths`（标好每项 `type: sft|rollout`）、`data.robot_type`、`data.model_type`。`run_compute_advantages_ensemble.sh` 默认用全部 GPU 起 torchrun，可加 `--nproc N` 或 `advantage.batch_size=...`。

6. **`num_nodes` 与卡数匹配**：配置注释"8卡用1，16卡用2"。`micro_batch_size` 同理（8卡32 / 16卡16）。按实际硬件改。

7. **Step3 的 `asset_id=restock_cola_recap` 与 `openpi.config_name=restock_cola_sm2sm`** 需要对应的 OpenPI norm-stats 资产存在；`config_name` 已在 `rlinf/models/embodiment/openpi/dataconfig/__init__.py` 注册（restock_cola/checkout_chips/fold_towel 的 `*_sm2sm` 均有）。若报 asset 找不到，需确认 assets 目录。

8. **环境（step0）正确**：`source /mnt/public/guqiuyi/RLinf_qiuyi/.venv/bin/activate` + `export PYTHONPATH=...:$PYTHONPATH`。三个 `run_*.sh` 内部还会尝试 `source switch_env openpi`（找不到则用当前环境，无碍）。已确认仓库内引用的 backbone（`pretrained_models/siglip2-so400m-patch14-224`、`gemma-3-270m`）、π0 base（`pi0_base_pytorch`）、以及示例数据集路径在磁盘上存在。

### ✅ 结论
- 你的 **step0 / step1 / step2 / step3 顺序与脚本/配置对应关系都正确**。
- **唯一结构性缺失**：step1 与 step2 之间要插入 `merge_steam_ensemble.py` 合并出 `ensemble_size=3` 的 checkpoint（除非你改成单次 `ensemble_size=3` 训练）。
- 其余为"对齐类"注意事项：跑多 seed 要换 `experiment_name`、三步数据集对准同一任务、`advantage_tag`/`k`/`camera_keys` 对齐、填好 Step2 占位路径。

---

## 7. 关键文件索引

| 模块 | 路径 |
|---|---|
| 分箱数学 | `rlinf/data/datasets/steam/binning.py` |
| 帧对数据集/collator | `rlinf/data/datasets/steam/pair_dataset.py` |
| 混合数据集 | `rlinf/data/datasets/steam/mixture.py` |
| backbone（SigLIP+Gemma+MLP） | `rlinf/models/embodiment/steam/modeling_steam.py` |
| 单 critic | `rlinf/models/embodiment/steam/modeling_critic.py` |
| 集成 critic（min 聚合） | `rlinf/models/embodiment/steam/ensemble_modeling_critic.py` |
| 模型工厂 / checkpoint 资产 | `rlinf/models/embodiment/steam/__init__.py` |
| config / processor | `rlinf/models/embodiment/steam/configuration.py`, `processing.py` |
| Step1 worker | `rlinf/workers/sft/fsdp_steam_sft_worker.py` |
| 训练 runner | `rlinf/runners/sft_runner.py` |
| Step2 打标脚本 | `examples/steam/process/compute_advantages_ensemble.py` |
| 合并 ensemble | `examples/steam/process/merge_steam_ensemble.py` |
| 重打标 / 可视化 | `examples/steam/process/relabel_advantages.py`, `visualize_advantage.py` |
| 分布式工具 | `rlinf/data/process/distributed.py` |
| Step3 CFG worker | `rlinf/workers/sft/fsdp_cfg_worker.py` |
| CFG 模型 | `rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py` |
| CFG 数据集（注入 advantage） | `rlinf/data/datasets/recap/cfg_model.py` |
| OpenPI 任务配置 | `rlinf/models/embodiment/openpi/dataconfig/__init__.py` |
</content>
</invoke>
