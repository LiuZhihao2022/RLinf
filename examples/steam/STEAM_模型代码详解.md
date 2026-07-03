# STEAM 模型代码详解：backbone 与 critic

> 专讲两个文件：
> - `rlinf/models/embodiment/steam/modeling_steam.py` —— **网络主体 `SteamBackbone`**（SigLIP + Gemma3 + 融合 MLP）
> - `rlinf/models/embodiment/steam/modeling_critic.py` —— **对外封装 `SteamCriticModel`**（接观测、算 loss、出 advantage）
>
> 配合阅读：`modeling_steam.py` 只负责"把一对图像+语言 → 一个融合特征向量"；`modeling_critic.py` 负责"在融合特征上加分类头、算交叉熵、把 bin 概率换成 `[-1,1]` 的有符号 advantage"。两者是**主体 / 外壳**的关系。

---

## 0. 先建立全局直觉

STEAM 要回答的问题不是"这一帧的价值是多少"，而是：

> 给定**两帧** `(frame_t, frame_{t+k})` 和语言指令 `ℓ`，这一对到底是**前进（progress）**还是**后退（regress）**？前进/后退到什么程度？

所以它本质是一个**帧对分类器**，输出 `num_bins` 个 bin 上的分布（bin 的左半是后退、右半是前进）。这决定了三个设计：

1. **两帧要分别编码、再按顺序拼接**（不能跨帧 pool）——否则 `(t, t+k)` 和 `(t+k, t)` 会变得一样，方向信息丢失。
2. **head 输出 `num_bins` 维**，做 `num_bins` 路交叉熵；二分类是 `num_bins=2` 的退化。
3. 把分布**坍缩成一个有符号标量** `∈[-1,1]` 当 advantage（正=前进，负=后退，绝对值=强度）。

数据全程的形状主线（记住这条就不会乱）：

```
观测 dict ──_stack_observation──▶ (input_ids, attn, images, image_mask)
images: [B, num_cameras, num_frames, 3, H, W]      # num_frames = num_frames_per_pair = 2 (即 t 和 t+k)
        │
        │  SteamBackbone._compute_projected_features
        ▼
fused:  [B, fusion_hidden_dim * (num_frames + 1)]   # 两帧图像特征 + 一个语言特征，拼起来
        │
        │  value_head (Linear→GELU→Dropout→Linear)
        ▼
logits: [B, num_bins]
        │  softmax
probs:  [B, num_bins]
        │  _predicted_signed_value
predicted_values: [B]   ∈ [-1, 1]    # 这就是单模型的 advantage
```

---

## 1. `modeling_steam.py` —— 网络主体 `SteamBackbone`

文件职责（见文件头 docstring，`modeling_steam.py:15-40`）：拥有"判断一对帧是 progress 还是 regress"的**神经网络结构**，但**不算 loss、不打包输出**——那是 critic 的事。对外只暴露一个核心方法 `_compute_projected_features`，返回融合特征。

### 1.1 `__init__`：搭了哪些零件（`modeling_steam.py:245-320`）

按顺序构造 5 类组件：

| 组件 | 代码 | 作用 | 关键点 |
|---|---|---|---|
| **vision_encoder** | `AutoModel.from_pretrained(cfg.vision_repo_id)` (`:255`) | SigLIP，把一张图 → 一个池化向量 | 用 HF AutoModel 加载，dtype = `_resolve_load_dtype(cfg.dtype)` |
| **language_model** | `_load_language_model(cfg.language_repo_id)` (`:260`) | Gemma-3-270M，把 prompt token → 隐藏序列 | 见 §1.2 |
| **image_projector** | `Linear(vision_feat_dim → fusion_hidden_dim) + GELU + Dropout` (`:289`) | 视觉特征投到统一融合维度 | **所有帧、所有相机共享同一个 projector** |
| **language_projector** | `Linear(lang_feat_dim → fusion_hidden_dim) + GELU + Dropout` (`:294`) | 语言特征投到同一融合维度 | |
| **value_head** | `Linear(fused_dim → fusion_hidden_dim) + GELU + Dropout + Linear(→ num_bins)` (`:307`) | 在融合特征上做 `num_bins` 路分类 | `head_out_dim = cfg.num_bins`（默认 2） |

还做了两件准备工作：

- **图像归一化常数**（`:266-283`）：从 SigLIP 的 image processor 读出原生分辨率 `image_resolution`（如 224×224 或 384×384）和 `image_mean/image_std`，注册成 buffer。**注意**：真正的 resize/归一化在 forward 里做（见 §1.4），这里只是把均值方差存好。
- **融合维度**：`fused_dim = fusion_hidden_dim * (num_frames_per_pair + 1)`（`:300`）。`+1` 是给语言特征留的位置。例如 `fusion_hidden_dim=512, num_frames_per_pair=2` → `fused_dim = 512*3 = 1536`。

可选开关（`:314-320`）：`use_gradient_checkpointing`（给 vision/language 开 checkpoint 省显存）、`freeze_vision_encoder` / `freeze_language_model`（冻结对应 backbone）。

### 1.2 加载语言模型的小技巧 `_load_language_model`（`:192-225`）

Gemma 这类模型的 HF 类是 `*ForCausalLM`（带 LM head）。但 STEAM **不需要 LM head**，只要 transformer 主体输出隐藏态。所以：

- 若架构名以 `ForCausalLM` 结尾 → 用 `AutoModelForCausalLM` 加载，然后取 `.model`（剥掉 LM head）。
- 否则 → 直接 `AutoModel`。

这样拿到的是"纯 encoder 主体"，后面对它的 `last_hidden_state` 做 mean-pool。

### 1.3 核心前向 `_compute_projected_features`（`:426-512`）

这是整个 backbone 的主函数。输入/输出：

```python
def _compute_projected_features(input_ids, attention_mask, images, image_attention_mask):
    # input_ids/attention_mask: [B, T]        语言 token 和 mask
    # images:               [B, Nc, Nf, 3, H, W]   Nc=相机数, Nf=帧数(=2)
    # image_attention_mask: [B, Nc, Nf] bool       哪些(相机,帧)槽位是真的
    return fused, per_frame_image_features, language_feature
    # fused:                    [B, fusion_hidden_dim * (Nf+1)]
    # per_frame_image_features: [B, Nf, fusion_hidden_dim]
    # language_feature:         [B, fusion_hidden_dim]
```

内部 8 步（每步标了形状）：

1. **形状校验** `_check_shapes`（`:454`）：确认 `images.ndim==6`、帧数 == `num_frames_per_pair` 等。
2. **有效性校验**（`:460-467`）：每个样本至少要有一个有效 (相机,帧) 槽位、至少一个有效语言 token，否则报错。
3. **拉平跑视觉**（`:471-475`）：把 `[B, Nc, Nf, 3, H, W]` reshape 成 `[B·Nc·Nf, 3, H, W]`，一次性过 vision encoder（比逐张跑快）。先做归一化 `_apply_siglip_normalisation`（见 §1.4）。
4. **视觉编码**（`:477-479`）：`vision_feats = _encode_vision(flat_images)` → `[B·Nc·Nf, vision_feat_dim]`。若 `freeze_vision_encoder` 则包在 `torch.no_grad()` 里。
5. **语言编码**（`:481-485`）：`_encode_prompt` → `[B, lang_feat_dim]`（见 §1.5）。
6. **投影 + 还原形状**（`:490-497`）：`image_projector(vision_feats)` → `[B·Nc·Nf, D]`，再 view 回 `[B, Nc, Nf, D]`（`D=fusion_hidden_dim`）。
7. **逐帧、按相机 mean-pool（带 mask）**（`:499-505`）——**关键**：
   ```python
   cam_mask_float = image_attention_mask.unsqueeze(-1)        # [B, Nc, Nf, 1]
   projected_masked = projected * cam_mask_float              # 缺失相机置 0
   cam_counts = cam_mask_float.sum(dim=1).clamp_min(1.0)      # [B, Nf, 1] 每帧有效相机数
   per_frame_features = projected_masked.sum(dim=1) / cam_counts  # [B, Nf, D]
   ```
   即：**同一帧的多个相机视角取平均**（缺失视角不参与、不稀释），得到每帧一个特征。注意 pool 的是**相机轴**，**帧轴保留**（这就是为什么方向信息不丢）。
8. **语言投影 + 融合**（`:507-512`）：`language_projector(lang)` → `[B, D]`，再 `self._fuse(per_frame_features, lang)`。

### 1.4 图像归一化 `_apply_siglip_normalisation`（`:326-365`）

collator 已经把图像 resize 到原生分辨率、转成 `[0,1]` float，所以这里**主要只做 SigLIP 的 `(x-mean)/std`**。但为了鲁棒，还加了"退化输入兜底"：

- `uint8` → `/255`；
- 检测到 `[-1,1]` 范围 → `(x+1)/2`；检测到 `[0,255]` 范围 → `/255`；
- 若分辨率不对 → `F.interpolate` 双线性 resize 到 `image_resolution`；
- 最后 `(x - image_mean) / image_std`。

### 1.5 语言编码 `_encode_prompt`（`:386-398`）

```python
hidden = language_model(input_ids, attention_mask).last_hidden_state   # [B, T, lang_dim]
mask = attention_mask.unsqueeze(-1)
language_feature = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)  # [B, lang_dim] 掩码均值池化
```
即对 token 维做 **mask 加权平均**（padding token 不计入）。

### 1.6 融合 `_fuse`（`:404-424`）

```python
frames_concat = per_frame_image_features.reshape(B, -1)   # [B, Nf*D]  两帧特征首尾相接(保留顺序)
fused = torch.cat([frames_concat, language_feature], -1)  # [B, Nf*D + D] = [B, (Nf+1)*D]
return self.fusion_norm(fused)                            # LayerNorm
```
**这一步是 STEAM 区别于普通 value 模型的灵魂**：两帧特征是**按 `(t, t+k)` 顺序 concat**，再接语言特征，最后 LayerNorm。顺序拼接让模型能区分"前进"与"后退"。

### 1.7 一句话总结 backbone

> 两帧 × 多相机的图 → 各自过 SigLIP → 投影 → 同帧多相机 mean-pool（保留帧轴）→ 两帧特征按顺序拼 + 语言特征拼 → LayerNorm → 融合向量 `[B, (Nf+1)·D]`。**到此为止还没有分类头、没有 loss。**

---

## 2. `modeling_critic.py` —— 对外封装 `SteamCriticModel`

文件职责（docstring `modeling_critic.py:15-42`）：在 backbone 的融合特征上**加分类头、算交叉熵、把概率换成有符号 advantage**，并提供与其它 value 模型一致的对外接口（`forward / predict / predict_value / from_checkpoint`），这样 FSDP worker 和离线打标脚本只靠 `model_type` 就能统一调度。

### 2.1 它持有什么（`__init__`, `:162-172`）

```python
self.model = SteamBackbone(config)   # ← 就是 §1 的主体；注意 value_head 其实在 backbone 里
self.label_smoothing = config.label_smoothing
```
> 命名上有个容易混的点：**分类头 `value_head` 定义在 `SteamBackbone` 内**（`modeling_steam.py:307`），critic 通过 `self.model.value_head` 调它。critic 自己不再额外建头。

### 2.2 观测契约 + 适配器 `_stack_observation`（`:225-283`）

critic 接收的 `observation` 是 collator 产出的 dict：

```python
observation = {
    "images":       {cam_name: Tensor[B, Nf, 3, H, W]},   # 每个相机一个 key, 帧轴在 tensor 里
    "image_masks":  {cam_name: Tensor[B, Nf] 或 [B]},      # 可选
    "tokenized_prompt":      Tensor[B, T],
    "tokenized_prompt_mask": Tensor[B, T],
}
```

`_stack_observation` 把它转成 backbone 要的 4 个张量：

- 相机按 key 排序后 `torch.stack(..., dim=1)` → `images: [B, Nc, Nf, 3, H, W]`（**相机轴是 dict 的 key，帧轴在每个 tensor 里**）。
- mask 若是 `[B]` 会广播成 `[B, Nf]`，再 stack 成 `image_mask: [B, Nc, Nf]`；缺失相机默认全 1。
- 返回 `(input_ids[B,T], attention_mask[B,T], images, image_mask)`。

### 2.3 损失 `_compute_loss`（`:289-345`）

`num_bins` 路交叉熵（覆盖二分类和多 bin）：

```python
loss = F.cross_entropy(logits,            # [B, num_bins]
                       bin_labels,        # [B], long, ∈ [0, num_bins)
                       label_smoothing=self.label_smoothing,
                       reduction="none")  # 逐样本
```
顺带算两个指标（放进 `CriticOutput` 对齐字段）：
- `acc_best`：argmax 命中精确 bin 的比例；
- `acc_neighbor`：`|pred_bin - target_bin| ≤ 1` 的比例（二分类时恒为 1）；
- `mae`：这里恒 0（此模式没有标量回归目标，只为字段对齐保留）。

> bin label 怎么来的？由数据侧 `pair_dataset` + `binning._signed_stride_to_bin` 把"有符号 stride"映射成 `[0, num_bins)` 的 long。critic 这边只管吃 bin idx 做分类。

### 2.4 **核心数学**：`_predicted_signed_value`（`:347-383`）

这是最容易卡住的地方，单独讲透。目标：把 `num_bins` 维概率 `probs` 坍缩成一个 `[-1,1]` 的有符号标量。

**bin 的有符号坐标**（`half = num_bins // 2`）：

```
bin 索引 b:   0      1     ...  half-1 | half  ...  num_bins-1
有符号值  : -half  -half+1 ...   -1   |  +1   ...   +half
```
代码（`:377-382`）：
```python
arange = torch.arange(num_bins)
signed_bin = where(arange < half, arange - half,        # [0,half)  → [-half, -1]
                                  arange - half + 1.0)  # [half,N)  → [+1, +half]
return (probs * signed_bin).sum(-1) / half   # 期望 / half，落进 [-1, 1]
```

即 **`E[signed_bin] / half`**：
- 左半 bin（后退）贡献负值，右半 bin（前进）贡献正值；
- 越靠两端（强前进/强后退）权重越大 → 标量同时编码**方向**和**强度**；
- 除以 `half` 把原本 `[-half, half]` 的期望归一到 `[-1, 1]`。

**二分类退化**（`num_bins=2, half=1`）：`signed_bin = [-1, +1]`，于是
`predicted_value = -p[0] + p[1] = 2·P(progress) − 1`。这正是报告里反复出现的那个式子。

> 直觉数值例子：`num_bins=4, half=2`，`signed_bin=[-2,-1,+1,+2]`。若 `probs=[0,0,0,1]`（完全确信最强前进）→ `(+2)/2 = +1`；若 `probs=[1,0,0,0]` → `(-2)/2 = -1`；若 `probs=[0,0.5,0.5,0]` → `(-1·0.5 + 1·0.5)/2 = 0`（前后退势均力敌）。

### 2.5 前向 `forward`（`:389-441`）与推理 `predict`（`:443-479`）

两者结构几乎一样，区别是 `predict` 用 `@torch.no_grad()` 且不算 loss：

```python
# 1. 观测 → 张量
input_ids, attn, images, image_mask = self._stack_observation(observation)
# 2. 主体出融合特征
hidden_states, _, _ = self.model._compute_projected_features(input_ids, attn, images, image_mask)  # [B, (Nf+1)*D]
# 3. 分类头
logits = self.model.value_head(hidden_states)   # [B, num_bins]
probs  = softmax(logits)                         # [B, num_bins]
# 4. 概率 → 有符号 advantage
predicted_values = self._predicted_signed_value(probs)   # [B] ∈ [-1,1]
# 5. (仅 forward) 若给了 labels 就算 loss
if labels is not None:
    expert_loss, metrics = self._compute_loss(logits, labels)
return CriticOutput(loss=..., predicted_values=..., logits=..., probs=..., ...)
```

`predict_value`（`:481-488`）就是 `predict(observation).predicted_values` 的快捷方式，返回 `[B]` 的有符号值（**注意是 `2P−1` 形态，不是 `P(progress)` 本身**）。

注意 `value_head` 前有个 dtype 对齐（`_module_parameter_dtype`，`:409`）：FSDP 混合精度下 head 权重可能是 bf16，所以先把 `hidden_states` cast 到 head 的 dtype 再喂，避免 matmul dtype 不一致。

### 2.6 `CriticOutput`（`:102-141`）

一个 `ModelOutput` dataclass，字段刻意与其它 value critic 对齐（duck-typing）。关键字段：
- `predicted_values: [B]` —— 有符号 advantage（单模型）；
- `logits / probs: [B, num_bins]`；
- `loss / expert_loss`、`cat_acc_best / cat_acc_neighbor / mae`、`hidden_states`、`progress_values`；
- `atoms` 恒 `None`（那是 categorical-value 的概念，这里不用）。
- **集成专有的 `member_*`、`prediction_mean/min/variance` 不在这里**——它们在 `EnsembleCriticOutput`（见 §3）。

### 2.7 从 checkpoint 重建 `from_checkpoint`（`:490-629`）

离线打标（Step 2）就靠它。做四件事：

1. 用 `cfg_dict={"model_path": ckpt}` + 可选 override 调 `get_model(cfg)`（`steam/__init__.py`），**按 `config.json` 的 `ensemble_size` 返回单模型或集成**。
2. 解析 tokenizer 来源（`_resolve_tokenizer_source`：优先显式路径 → checkpoint 内的 tokenizer 文件 → config 里的 `language_repo_id`）。
3. 加载/构造 `SteamImageProcessor`（分辨率要与 vision encoder 原生分辨率匹配）。
4. `attach_runtime_assets(processor, device)` 挂上 processor 和 device，`.to(device).eval()`。

---

## 3. 两文件之外：集成是怎么接上的（半页补充）

`SteamCriticModel` 是**单个 member**。集成 worst-of-N 由 `ensemble_modeling_critic.py::EnsembleSteamCriticModel` 包一层：

- 它持有 `members = nn.ModuleList([SteamCriticModel, ...])`；
- **训练**：`forward(obs, labels, member_idx=m)` 只跑第 m 个 member（worker 外层循环逐 member 喂独立随机 batch）；
- **推理 `predict`**：对每个 member 取 `predicted_values`，`member_predicted_values.min(dim=0)` 逐样本取**最小**（最保守/最"退步"的那个 member）作为聚合 advantage `prediction_min`，并把对应 member 的 logits/probs 也 gather 出来（保证 `signed_value(probs) == prediction_min`）。这就是论文 Eq.5 的 `A_STEAM = min_m A_m`。

所以调用栈是：`EnsembleSteamCriticModel.predict → 每个 SteamCriticModel.predict → SteamBackbone._compute_projected_features + value_head`。

---

## 4. 调用关系速查图

```
观测 dict
  │
SteamCriticModel.forward / predict           (modeling_critic.py)
  │  _stack_observation        dict → (ids, attn, images[B,Nc,Nf,3,H,W], mask)
  │
  ├─ self.model._compute_projected_features  (modeling_steam.py / SteamBackbone)
  │     ├─ _apply_siglip_normalisation  归一化
  │     ├─ _encode_vision               SigLIP → [B·Nc·Nf, vis_dim]
  │     ├─ image_projector + 同帧相机 mean-pool → per_frame [B,Nf,D]
  │     ├─ _encode_prompt               Gemma mask-pool → [B, lang_dim]
  │     ├─ language_projector           → [B, D]
  │     └─ _fuse                        两帧+语言 concat → LayerNorm → [B,(Nf+1)*D]
  │
  ├─ self.model.value_head              [B,(Nf+1)*D] → logits [B,num_bins]
  ├─ softmax → probs [B,num_bins]
  ├─ _predicted_signed_value            probs → advantage [B] ∈[-1,1]
  └─ (forward+labels) _compute_loss     num_bins 路交叉熵 → CriticOutput
```

---

## 5. 关键超参对结构的影响（速记）

| config 字段 | 影响的结构 | 说明 |
|---|---|---|
| `num_frames_per_pair`（=2） | 融合维度 `(Nf+1)*D`、`_check_shapes` | 标准配方恒为 2（即 t 和 t+k） |
| `num_bins`（=32） | `value_head` 输出维、loss 类数、`_predicted_signed_value` 的 bin 坐标 | 必须偶数；2 为二分类退化 |
| `fusion_hidden_dim`（=512） | 所有 projector 与 head 的隐藏宽度 | |
| `label_smoothing`（=0.05） | `_compute_loss` 的交叉熵 | |
| `freeze_vision_encoder / freeze_language_model` | backbone 是否回传梯度 | 冻结时对应分支包 `no_grad` |
| `precision / dtype` | 加载与前向 dtype | 默认 fp32 master + bf16 前向（FSDP） |
| `ensemble_size` | 由 `get_model` 决定返回单模型还是集成 | >1 时走 `EnsembleSteamCriticModel` |

---

## 6. 三个最易误解点（划重点）

1. **`value_head` 不在 critic 里，在 backbone 里**。critic 通过 `self.model.value_head` 用它。
2. **`predicted_values` 不是 `P(progress)`，是 `2P−1` 形态的有符号值 `∈[-1,1]`**（多 bin 时是有符号期望）。下游阈值/分位都基于这个 `[-1,1]` 量，不是概率。
3. **相机轴被 mean-pool，帧轴被保留并 concat**。方向信息靠"帧轴顺序拼接"承载——这是 STEAM 能区分前进/后退的根本。
</content>
