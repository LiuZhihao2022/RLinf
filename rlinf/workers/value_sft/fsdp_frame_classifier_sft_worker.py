# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP SFT worker for :class:`SuccessFailClassifier`.

Structurally parallel to :class:`FSDPBinaryValueSftWorker` (ARM + ReWiND
binary value critic) but specialised for a single-frame success/fail
classifier:

    * :class:`FrameClassifierDataset` — one sample per LeRobot frame,
      labels broadcast from per-episode ``is_success``.
    * :class:`FrameClassifierMixtureDataset` — optional weighted mixture
      across multiple LeRobot sources.
    * :class:`FrameClassifierDataCollator` — single-frame collator;
      produces ``observation = {images, image_masks, state?}`` plus
      ``labels: Tensor[B] float32`` (no tokenized prompt).
    * Loss:
      ``F.binary_cross_entropy_with_logits(logits, y_smooth,
        pos_weight=...)``
      with ``label_smoothing`` baked into the model and ``pos_weight``
      resolved per step according to ``data.pos_weight_mode``:

        - ``"auto"``     — global ``n_success / n_fail`` from dataset scan
                           (default)
        - ``"none"``     — disable rebalance (equivalent to 1.0)
        - ``<float>``    — user-specified scalar
        - ``"batch_adaptive"`` — per-batch ratio with fallback to the
                                 ``"auto"`` cached value on single-class
                                 batches

    Sampling stays on the standard ``DistributedSampler`` path — no
    ``WeightedRandomSampler``, which would double-random against
    :class:`_MixtureBase` and break FSDP rank-aware sharding (see the
    plan's §6 for the full rationale).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

os.environ["LIBAV_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
logging.getLogger("libav").setLevel(logging.ERROR)
logging.getLogger("av").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

import torch  # noqa: E402
from omegaconf import DictConfig, open_dict  # noqa: E402
from torch import Tensor  # noqa: E402

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager  # noqa: E402
from rlinf.models import get_model  # noqa: E402
from rlinf.scheduler import Worker  # noqa: E402
from rlinf.utils.distributed import all_reduce_dict  # noqa: E402


class _FrameDataLoaderImpl:
    """Thin wrapper so ``SFTRunner.set_global_step`` can swap epochs."""

    def __init__(self, data_loader: torch.utils.data.DataLoader) -> None:
        self._data_loader = data_loader

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        sampler = getattr(self._data_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        dataset = getattr(self._data_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def __iter__(self):
        yield from self._data_loader


def _frame_dataset_camera_shapes(dataset) -> dict[str, tuple[int, ...] | None]:
    """Return representative raw camera shapes from the dataset's first sample.

    Parallels :func:`_pair_dataset_camera_shapes` in the binary-value worker:
    reads one sample, returns ``{camera_key: shape | None}`` so a multi-
    dataset mixture can be shape-checked before training starts.
    """
    sample = dataset[0]
    shapes: dict[str, tuple[int, ...] | None] = {}
    images = sample.get("images", {})
    for camera_key, image in images.items():
        shapes[camera_key] = (
            tuple(int(dim) for dim in image.shape) if image is not None else None
        )
    for camera_key in sample.get("image_masks", {}).keys():
        shapes.setdefault(camera_key, None)
    return shapes


def _validate_train_dataset_shapes(
    named_datasets: list[tuple[str, Any]],
) -> None:
    """Fail fast when a train mixture combines incompatible raw image sizes.

    Mirrors :func:`rlinf.workers.value_sft.fsdp_binary_value_sft_worker.\
_validate_train_dataset_shapes`. :class:`FrameClassifierDataCollator` per-
    camera stack requires identical raw ``(H, W)`` across all samples in a
    batch; without this preflight a mixture with heterogeneous resolutions
    would crash at the first mismatched batch instead of at startup.
    """
    if len(named_datasets) <= 1:
        return

    reference_name, reference_dataset = named_datasets[0]
    reference_shapes = _frame_dataset_camera_shapes(reference_dataset)
    mismatches: list[str] = []

    for dataset_name, dataset in named_datasets[1:]:
        shapes = _frame_dataset_camera_shapes(dataset)
        differing = {
            camera_key: (reference_shapes.get(camera_key), shapes.get(camera_key))
            for camera_key in sorted(set(reference_shapes) | set(shapes))
            if reference_shapes.get(camera_key) != shapes.get(camera_key)
        }
        if differing:
            mismatches.append(
                f"{dataset_name}: {differing} (reference {reference_name}: "
                f"{reference_shapes})"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Frame classifier train mixtures require identical raw camera "
            "shapes across all train_data_paths because "
            "FrameClassifierDataCollator does not resize images inside the "
            f"collator. Incompatible shapes: {mismatch_text}"
        )


def _ensure_precision_cfg(model_cfg: DictConfig) -> str:
    """Default unset precision to fp32 (matches binary_value_sft rationale)."""
    precision = getattr(model_cfg, "precision", None)
    if precision not in (None, "", "null"):
        return str(precision)
    with open_dict(model_cfg):
        model_cfg.precision = "fp32"
    logger.warning(
        "[FrameClassifierSFT] actor.model.precision was unset; defaulting to "
        "fp32 for FSDP stability. Forward compute dtype is still controlled by "
        "fsdp_config.mixed_precision.param_dtype."
    )
    return str(model_cfg.precision)


def _parse_pos_weight_mode(raw: Any) -> Union[str, float]:
    """Parse ``data.pos_weight_mode`` into one of {auto, none, batch_adaptive, float}.

    Accepts:
        * ``None`` / unset → ``"auto"`` (default)
        * strings ``"auto"`` / ``"none"`` / ``"batch_adaptive"``
        * any numeric convertible value → ``float(value)``

    Raises ``ValueError`` for anything else, per 78bc04dd fail-loud policy.
    """
    if raw is None:
        return "auto"
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        if value <= 0.0:
            raise ValueError(
                f"data.pos_weight_mode must be positive when given as a "
                f"scalar; got {value}"
            )
        return value
    if isinstance(raw, str):
        txt = raw.strip().lower()
        if txt in {"auto", "none", "batch_adaptive"}:
            return txt
        try:
            value = float(txt)
        except ValueError as err:
            raise ValueError(
                f"data.pos_weight_mode must be one of 'auto', 'none', "
                f"'batch_adaptive', or a positive float; got {raw!r}"
            ) from err
        if value <= 0.0:
            raise ValueError(
                f"data.pos_weight_mode as a scalar must be > 0; got {value}"
            )
        return value
    raise ValueError(
        f"data.pos_weight_mode has unsupported type {type(raw).__name__} "
        f"(value {raw!r})"
    )


def _collect_non_finite_tensor_paths(value: Any, prefix: str) -> list[str]:
    """Return dotted tensor paths whose values contain NaN/Inf.

    Verbatim copy of the helper in
    :mod:`rlinf.workers.value_sft.fsdp_binary_value_sft_worker` so this
    module does not depend on a private symbol there.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or torch.isfinite(value.detach()).all():
            return []
        return [prefix]
    if isinstance(value, dict):
        bad: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            bad.extend(_collect_non_finite_tensor_paths(child, child_prefix))
        return bad
    if isinstance(value, (list, tuple)):
        bad = []
        for idx, child in enumerate(value):
            bad.extend(_collect_non_finite_tensor_paths(child, f"{prefix}[{idx}]"))
        return bad
    return []


class FSDPFrameClassifierSftWorker(FSDPModelManager, Worker):
    """FSDP SFT worker for :class:`SuccessFailClassifier`.

    Reads ``data.train_data_paths`` / ``data.eval_data_paths`` (the same
    config shape as the binary-value worker) and feeds one frame per
    batch through ``model(observation=obs, labels=labels,
    pos_weight=...)``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        self._pos_weight_mode: Union[str, float] = _parse_pos_weight_mode(
            cfg.data.get("pos_weight_mode", "auto")
        )
        self._pos_weight_global: Optional[Tensor] = None

        self.data_loader, self.eval_data_loaders = self.build_dataloader()
        self.data_iter = iter(self.data_loader)

    def init_worker(self) -> None:
        self.setup_model_and_optimizer()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> torch.nn.Module:
        _ensure_precision_cfg(self.cfg.actor.model)
        return get_model(self.cfg.actor.model)

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save weights + classifier config + image processor alongside."""
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            from rlinf.models.embodiment.success_fail_classifier import (
                save_success_fail_classifier_checkpoint_assets,
            )

            save_success_fail_classifier_checkpoint_assets(
                save_path=save_path,
                cfg=self.cfg.actor.model,
                processor=getattr(self, "processor", None),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def build_dataloader(self):
        """Build train + eval dataloaders from ``data.*`` config."""
        try:
            import av

            av.logging.set_level(av.logging.ERROR)
        except (ImportError, AttributeError):
            pass

        from rlinf.data.datasets.cfg.success_fail_dataset import (
            DEFAULT_CAMERA_KEYS,
            FrameClassifierDataset,
            FrameClassifierMixtureDataset,
            build_frame_classifier_loader,
        )
        from rlinf.models.embodiment.success_fail_classifier import (
            FrameClassifierDataCollator,
            SuccessFailClassifierImageProcessor,
        )

        data_cfg = self.cfg.get("data", {})
        model_cfg = self.cfg.actor.model

        camera_keys = tuple(data_cfg.get("camera_keys", DEFAULT_CAMERA_KEYS))
        state_key = str(data_cfg.get("state_key", "state"))
        use_proprio = bool(getattr(model_cfg, "use_proprio", True))
        max_state_dim = int(getattr(model_cfg, "max_state_dim", 32))
        image_size = int(getattr(model_cfg, "image_size", 224))
        robot_type_default = str(data_cfg.get("robot_type", "libero"))
        model_type_default = str(data_cfg.get("model_type", "pi05"))
        action_dim = int(data_cfg.get("action_dim", 32))
        default_prompt = data_cfg.get("default_prompt", None)

        processor = SuccessFailClassifierImageProcessor(
            image_size=image_size,
            image_keys=camera_keys,
        )
        self.processor = processor

        train_collator = FrameClassifierDataCollator(
            processor=processor,
            use_proprio=use_proprio,
        )
        eval_collator = FrameClassifierDataCollator(
            processor=processor,
            use_proprio=use_proprio,
        )

        data_root = data_cfg.get("data_root", None)

        def _resolve(path: str) -> str:
            if data_root and not os.path.isabs(path):
                return os.path.join(data_root, path)
            return path

        def _build_dataset(entry: dict) -> FrameClassifierDataset:
            entry = dict(entry)
            ds_path = _resolve(entry["dataset_path"])
            dataset_type = str(
                entry.get("type", data_cfg.get("dataset_type", "rollout"))
            ).lower()
            robot_type = str(entry.get("robot_type", robot_type_default))
            model_type = str(entry.get("model_type", model_type_default))

            # Only the "rollout" path needs an is_success scan; "sft" trusts
            # the dataset to be all-success. Defaulting to rollout reflects
            # the classifier's training use case (needs fail data).
            include_success = bool(
                entry.get(
                    "include_success",
                    data_cfg.get("include_success", True),
                )
            )
            include_fail = bool(
                entry.get(
                    "include_fail",
                    data_cfg.get("include_fail", True),
                )
            )

            return FrameClassifierDataset(
                dataset_path=ds_path,
                dataset_type=dataset_type,
                camera_keys=camera_keys,
                include_state=use_proprio,
                state_key=state_key,
                max_state_dim=max_state_dim,
                robot_type=robot_type,
                model_type=model_type,
                action_dim=action_dim,
                default_prompt=default_prompt,
                norm_stats_dir=entry.get(
                    "norm_stats_dir", data_cfg.get("norm_stats_dir", None)
                ),
                asset_id=entry.get("asset_id", data_cfg.get("asset_id", None)),
                include_success=include_success,
                include_fail=include_fail,
                min_episode_length=int(
                    entry.get(
                        "min_episode_length",
                        data_cfg.get("min_episode_length", 1),
                    )
                ),
                source_name=ds_path,
            )

        train_entries = [
            dict(e)
            for e in data_cfg.get("train_data_paths", [])
            if e.get("dataset_path")
        ]
        if not train_entries:
            raise ValueError(
                "data.train_data_paths must contain at least one entry "
                "with 'dataset_path'."
            )

        datasets_with_weights: list[tuple[FrameClassifierDataset, float]] = []
        named_train_datasets: list[tuple[str, FrameClassifierDataset]] = []
        for entry in train_entries:
            dataset = _build_dataset(entry)
            weight = float(entry.get("weight", 1.0))
            resolved_path = _resolve(entry["dataset_path"])
            datasets_with_weights.append((dataset, weight))
            named_train_datasets.append((resolved_path, dataset))
            logger.info(
                "[FrameClassifierSFT] Loaded train dataset: %s (%d frames, weight=%.4f)",
                resolved_path,
                len(dataset),
                weight,
            )

        # Fail fast on heterogeneous raw camera shapes — otherwise the first
        # mismatched batch crashes inside FrameClassifierDataCollator mid-
        # training with an opaque ValueError.
        _validate_train_dataset_shapes(named_train_datasets)

        balance_dataset_weights = bool(
            data_cfg.get(
                "balance_weights",
                data_cfg.get("balance_dataset_weights", True),
            )
        )

        if len(datasets_with_weights) == 1:
            train_dataset: torch.utils.data.Dataset = datasets_with_weights[0][0]
        else:
            train_dataset = FrameClassifierMixtureDataset(
                datasets=datasets_with_weights,
                mode="train",
                balance_dataset_weights=balance_dataset_weights,
                seed=int(data_cfg.get("seed", 42)),
            )

        logger.info(
            "[FrameClassifierSFT] Train: %d dataset(s), %d samples total",
            len(datasets_with_weights),
            len(train_dataset),
        )

        # Cache the global pos_weight (used by the "auto" mode as well as
        # the "batch_adaptive" fallback when a batch is single-class).
        global_pw_value = float(getattr(train_dataset, "pos_weight", 1.0))
        self._pos_weight_global = torch.tensor(
            global_pw_value, dtype=torch.float32, device=self.device
        )
        logger.info(
            "[FrameClassifierSFT] pos_weight_mode=%s, pos_weight_global=%.4f",
            self._pos_weight_mode,
            global_pw_value,
        )

        # --- DataLoader knobs ---
        pin_memory = bool(data_cfg.get("pin_memory", True))
        train_num_workers = int(data_cfg.get("train_num_workers", 0))
        eval_num_workers = int(data_cfg.get("eval_num_workers", train_num_workers))
        prefetch_factor = data_cfg.get("prefetch_factor", 2)
        persistent_workers = bool(data_cfg.get("persistent_workers", True))

        train_loader = build_frame_classifier_loader(
            train_dataset,
            batch_size=int(self.cfg.actor.micro_batch_size),
            collate_fn=train_collator,
            num_workers=train_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            world_size=self._world_size,
            rank=self._rank,
            shuffle=True,
            drop_last=True,
            seed=int(data_cfg.get("seed", 42)),
        )

        eval_entries = data_cfg.get("eval_data_paths", []) or []
        eval_data_loaders: list[tuple[str, _FrameDataLoaderImpl]] = []
        for entry in eval_entries:
            entry = dict(entry)
            if not entry.get("dataset_path"):
                continue
            ds = _build_dataset(entry)
            name = entry.get("name", Path(_resolve(entry["dataset_path"])).stem)
            eval_loader = build_frame_classifier_loader(
                ds,
                batch_size=int(self.cfg.actor.micro_batch_size),
                collate_fn=eval_collator,
                num_workers=eval_num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                world_size=self._world_size,
                rank=self._rank,
                shuffle=False,
                drop_last=False,
                seed=int(data_cfg.get("seed", 42)),
            )
            eval_data_loaders.append((name, _FrameDataLoaderImpl(eval_loader)))
            logger.info(
                "[FrameClassifierSFT] Eval '%s': %d samples", name, len(ds)
            )

        return _FrameDataLoaderImpl(train_loader), eval_data_loaders

    # ------------------------------------------------------------------
    # pos_weight + loss helpers
    # ------------------------------------------------------------------

    def _resolve_pos_weight(self, labels: Tensor) -> Optional[Tensor]:
        mode = self._pos_weight_mode
        if mode == "none":
            return None
        if isinstance(mode, float):
            return torch.tensor(mode, dtype=torch.float32, device=labels.device)
        if mode == "auto":
            return self._pos_weight_global
        if mode == "batch_adaptive":
            n_pos = float(labels.sum().item())
            n_neg = float(labels.numel() - n_pos)
            if n_pos <= 0.0 or n_neg <= 0.0:
                return self._pos_weight_global
            return torch.tensor(
                n_neg / n_pos, dtype=torch.float32, device=labels.device
            )
        raise RuntimeError(f"Unhandled pos_weight_mode: {mode!r}")  # unreachable

    def _label_pos_frac(self, labels: Tensor) -> float:
        """Fraction of fail-labelled samples in the batch."""
        return float(labels.to(dtype=torch.float32).mean().item())

    def _prepare_input(self, batch: dict) -> tuple[dict, Tensor]:
        """Move a collated batch to device, split into observation + labels."""

        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device, non_blocking=True)
            if isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            return x

        observation = _to_device(batch["observation"])
        labels = _to_device(batch["labels"])
        return observation, labels

    def _raise_if_non_finite(
        self,
        result,
        observation: dict,
        labels: Tensor,
        micro_idx: int,
        grad_accum: int,
    ) -> None:
        bad_paths: list[str] = []
        bad_paths.extend(_collect_non_finite_tensor_paths(observation, "observation"))
        bad_paths.extend(_collect_non_finite_tensor_paths(labels, "labels"))
        for field_name in ("loss", "predicted_values", "logits", "hidden_states"):
            field_value = getattr(result, field_name, None)
            bad_paths.extend(
                _collect_non_finite_tensor_paths(field_value, f"result.{field_name}")
            )
        if not bad_paths:
            return
        label_pos_frac = self._label_pos_frac(labels)
        global_step = int(getattr(self, "global_step", 0))
        precision = getattr(self.cfg.actor.model, "precision", None)
        unique_paths = ", ".join(sorted(set(bad_paths)))
        raise FloatingPointError(
            "Non-finite tensor detected during frame classifier SFT "
            f"(global_step={global_step}, micro_batch={micro_idx + 1}/{grad_accum}, "
            f"precision={precision}, label_pos_frac={label_pos_frac:.4f}). "
            f"Offending tensors: {unique_paths}."
        )

    def _fetch_next_batch(self) -> dict:
        try:
            return next(self.data_iter)
        except StopIteration:
            new_epoch = getattr(self, "_current_epoch", 0) + 1
            self._current_epoch = new_epoch
            self.data_loader.set_epoch(new_epoch)
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)

    # ------------------------------------------------------------------
    # Training / eval steps
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, result, labels: Tensor, pos_weight: Optional[Tensor]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if result.loss is not None:
            metrics["loss"] = float(result.loss.detach().item())
        if result.cat_acc_best is not None:
            metrics["accuracy"] = float(result.cat_acc_best.detach().item())
        if result.predicted_values is not None:
            probs = result.predicted_values.detach().float()
            metrics["prob_fail_mean"] = float(probs.mean().item())
            if probs.numel() >= 2:
                metrics["prob_fail_std"] = float(probs.std(unbiased=False).item())
            else:
                metrics["prob_fail_std"] = 0.0
        if result.logits is not None:
            logits = result.logits.detach().float().squeeze(-1)
            metrics["logit_mean"] = float(logits.mean().item())
            if logits.numel() >= 2:
                metrics["logit_std"] = float(logits.std(unbiased=False).item())
            else:
                metrics["logit_std"] = 0.0
        if labels is not None:
            metrics["label_pos_frac"] = self._label_pos_frac(labels)
        if pos_weight is not None:
            metrics["pos_weight"] = float(pos_weight.detach().item())
        else:
            metrics["pos_weight"] = 1.0
        return metrics

    def _backward_one_micro_batch(
        self, grad_accum: int, micro_idx: int
    ) -> dict[str, float]:
        backward_ctx = self.before_micro_batch(
            self.model, is_last_micro_batch=(micro_idx + 1) == grad_accum
        )
        batch = self._fetch_next_batch()
        observation, labels = self._prepare_input(batch)
        pos_weight = self._resolve_pos_weight(labels)

        with self.amp_context:
            result = self.model(
                observation=observation, labels=labels, pos_weight=pos_weight
            )
            loss = result.loss
        if loss is None:
            raise RuntimeError(
                "SuccessFailClassifier returned no loss during training."
            )
        self._raise_if_non_finite(
            result=result,
            observation=observation,
            labels=labels,
            micro_idx=micro_idx,
            grad_accum=grad_accum,
        )

        metrics = self._compute_metrics(result, labels, pos_weight)
        scaled_loss = loss / grad_accum
        with backward_ctx:
            self.grad_scaler.scale(scaled_loss).backward()
        return metrics

    @staticmethod
    def _mean_metrics(dicts: list[dict[str, float]]) -> dict[str, float]:
        agg: dict[str, list[float]] = {}
        for d in dicts:
            for k, v in d.items():
                agg.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in agg.items()}

    def run_training(self) -> dict[str, float]:
        """Execute one global training step."""
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            use_grad_ckpt = bool(
                getattr(self.cfg.actor.model, "use_gradient_checkpointing", False)
            ) or bool(
                getattr(self.cfg.actor.fsdp_config, "gradient_checkpointing", False)
            )
            if use_grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            micro_bs = self.cfg.actor.micro_batch_size
            global_bs = self.cfg.actor.global_batch_size
            assert global_bs % (micro_bs * self._world_size) == 0, (
                f"global_batch_size={global_bs} must be divisible by "
                f"micro_batch_size * world_size = {micro_bs * self._world_size}"
            )
            grad_accum = global_bs // micro_bs // self._world_size

            micro_metrics = [
                self._backward_one_micro_batch(grad_accum, i)
                for i in range(grad_accum)
            ]
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            train_metrics = self._mean_metrics(micro_metrics)
            train_metrics["grad_norm"] = float(grad_norm)
            train_metrics["lr"] = float(lr_list[0]) if lr_list else 0.0

            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )
            self.lr_scheduler.step()

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()
                    self.offload_optimizer()
            return train_metrics

    def _eval_batch(
        self, observation: dict, labels: Tensor
    ) -> dict[str, float]:
        pos_weight = self._resolve_pos_weight(labels)
        with self.amp_context:
            result = self.model(
                observation=observation, labels=labels, pos_weight=pos_weight
            )
        return self._compute_metrics(result, labels, pos_weight)

    def run_eval(self) -> dict[str, float]:
        if not self.eval_data_loaders:
            return {}
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)

            self.model.eval()
            per_dataset: dict[str, dict[str, float]] = {}
            with torch.no_grad():
                for ds_name, loader in self.eval_data_loaders:
                    batch_metrics: list[dict[str, float]] = []
                    for batch in loader:
                        observation, labels = self._prepare_input(batch)
                        batch_metrics.append(self._eval_batch(observation, labels))
                    if not batch_metrics:
                        continue
                    per_dataset[ds_name] = self._mean_metrics(batch_metrics)

            if not per_dataset:
                return {}

            final: dict[str, float] = {}
            for ds_name, metrics in per_dataset.items():
                for k, v in metrics.items():
                    final[f"{ds_name}/{k}"] = v

            all_keys: set[str] = set()
            for metrics in per_dataset.values():
                all_keys.update(metrics.keys())
            for k in sorted(all_keys):
                vals = [m[k] for m in per_dataset.values() if k in m]
                if vals:
                    final[k] = sum(vals) / len(vals)

            final = all_reduce_dict(final, op=torch.distributed.ReduceOp.AVG)

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()
            return final

    def set_global_step(self, step: int) -> None:
        """Rotate sampler epochs on exhaustion (mirrors binary_value worker)."""
        self.global_step = step
        loader_len = len(self.data_loader)
        if loader_len == 0:
            return
        grad_accum = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        steps_per_epoch = max(1, loader_len // grad_accum)
        new_epoch = step // steps_per_epoch
        current = getattr(self, "_current_epoch", -1)
        if current != new_epoch:
            self._current_epoch = new_epoch
            self.data_loader.set_epoch(new_epoch)
            self.data_iter = iter(self.data_loader)


__all__ = ["FSDPFrameClassifierSftWorker"]
