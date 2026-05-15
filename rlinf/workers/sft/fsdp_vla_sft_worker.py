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
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils import _pytree

from rlinf.config import SupportedModel
from rlinf.data.datasets.cfg.mixture_datasets import CfgMixtureDataset
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.cfg.utils import (
    AdvantagePreservingDataset,
    PositiveAdvantageOnlySubset,
    SftPlainDataLoaderImpl,
    cast_image_features,
    create_distributed_torch_dataloader,
    fix_episode_data_index,
    load_advantages_lookup,
    load_continuous_advantages_lookup,
    load_positive_threshold,
)
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type in [SupportedModel.OPENPI]:
            if self._uses_dataset_entries(data_paths):
                if eval_dataset:
                    raise NotImplementedError(
                        "eval is not supported for embodied OpenPI SFT with "
                        "list-style data.train_data_paths right now."
                    )
                return self._build_openpi_dataset_entries_dataloader(data_paths)

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                data_kwargs=self._openpi_data_kwargs(),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif self._is_lingbotvla_model(model_type):
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    @staticmethod
    def _uses_dataset_entries(data_paths: Any) -> bool:
        """Return True when train_data_paths is a list of dataset config entries."""
        if not isinstance(data_paths, (list, ListConfig)) or len(data_paths) == 0:
            return False
        first = data_paths[0]
        return isinstance(first, (dict, DictConfig)) and "dataset_path" in first

    @staticmethod
    def _is_lingbotvla_model(model_type: SupportedModel) -> bool:
        """Check LingBotVLA only when that optional enum member exists."""
        lingbotvla_model = getattr(SupportedModel, "LINGBOTVLA", None)
        return lingbotvla_model is not None and model_type == lingbotvla_model

    def _openpi_data_kwargs(self) -> Any:
        """Read OpenPI data overrides from the current and legacy config paths."""
        data_kwargs = getattr(self.cfg.actor.model, "openpi_data", None)
        if data_kwargs is None:
            data_kwargs = getattr(self.cfg.actor, "openpi_data", None)
        return data_kwargs

    def _build_openpi_dataset_entries_dataloader(self, datasets_config: Any):
        """Build OpenPI SFT dataloader from one or more LeRobot dataset entries."""
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        import openpi.shared.download as download
        import openpi.training.data_loader as openpi_data_loader
        import openpi.transforms as transforms
        from openpi.training import checkpoints as _checkpoints

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        data_cfg = self.cfg.get("data", {})
        openpi_cfg = self.cfg.actor.model.openpi
        advantage_tag = data_cfg.get("advantage_tag", None)
        awbc_cfg = data_cfg.get("awbc", {}) or {}
        awbc_enabled = bool(awbc_cfg.get("enabled", False))
        if awbc_enabled and not advantage_tag:
            raise ValueError("data.advantage_tag is required when data.awbc.enabled=True")
        advantage_filter = str(data_cfg.get("advantage_filter", "") or "").lower()
        if advantage_filter not in ("", "all", "positive_only"):
            raise ValueError(
                "data.advantage_filter must be unset, 'all', or 'positive_only'; "
                f"got {advantage_filter!r}."
            )
        positive_only = advantage_filter == "positive_only" and not awbc_enabled

        first_path = datasets_config[0]["dataset_path"]
        config = get_openpi_config(
            openpi_cfg.config_name,
            batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            repo_id=first_path,
            asset_id=openpi_cfg.get("asset_id", None),
            data_kwargs=self._openpi_data_kwargs(),
        )
        data_config = config.data.create(config.assets_dirs, config.model)

        norm_stats = data_config.norm_stats
        if norm_stats is None and data_config.asset_id is not None:
            checkpoint_dir = download.maybe_download(
                str(self.cfg.actor.model.model_path)
            )
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir,
                data_config.asset_id,
            )
        norm_stats = norm_stats or {}

        state_history_size = getattr(
            data_config,
            "state_history_size",
            getattr(config.data, "state_history_size", 0),
        )
        state_future_size = getattr(
            data_config,
            "state_future_size",
            getattr(config.data, "state_future_size", 0),
        )
        state_step = getattr(
            data_config, "state_step", getattr(config.data, "state_step", 1)
        )

        def build_delta_timestamps(fps: float) -> dict[str, list[float]]:
            delta_timestamps = {
                key: [t / fps for t in range(config.model.action_horizon)]
                for key in data_config.action_sequence_keys
            }
            if state_history_size > 0 or state_future_size > 0:
                delta_timestamps["state"] = [
                    t * state_step / fps
                    for t in range(-state_history_size, state_future_size + 1)
                ]
            return delta_timestamps

        datasets_with_weights = []
        awbc_total_samples = 0
        awbc_kept_samples = 0
        for ds_config in datasets_config:
            data_path = ds_config["dataset_path"]
            local_path = Path(data_path).absolute()
            episodes = ds_config.get("episodes")
            weight = float(ds_config.get("weight", 1.0))

            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
                local_path.name, root=local_path
            )
            base_dataset = lerobot_dataset.LeRobotDataset(
                local_path.name,
                root=local_path,
                episodes=episodes,
                delta_timestamps=build_delta_timestamps(dataset_meta.fps),
                download_videos=False,
            )
            base_dataset.hf_dataset = cast_image_features(base_dataset.hf_dataset)

            if episodes is not None:
                fix_episode_data_index(base_dataset, episodes)

            if data_config.prompt_from_task:
                base_dataset = openpi_data_loader.TransformedDataset(
                    base_dataset,
                    [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
                )

            transformed_dataset = openpi_data_loader.TransformedDataset(
                base_dataset,
                [
                    *data_config.repack_transforms.inputs,
                    *data_config.data_transforms.inputs,
                    transforms.Normalize(
                        norm_stats, use_quantiles=data_config.use_quantile_norm
                    ),
                    *data_config.model_transforms.inputs,
                ],
            )

            final_dataset = transformed_dataset
            if awbc_enabled:
                advantages_lookup = load_advantages_lookup(data_path, advantage_tag)
                continuous_advantages_lookup = load_continuous_advantages_lookup(
                    data_path, advantage_tag
                )
                positive_threshold = load_positive_threshold(data_path, advantage_tag)
                max_continuous_advantage = max(
                    continuous_advantages_lookup.values(),
                    default=positive_threshold,
                )
                final_dataset = AdvantagePreservingDataset(
                    base_dataset=base_dataset,
                    transformed_dataset=transformed_dataset,
                    advantages_lookup=advantages_lookup,
                    continuous_advantages_lookup=continuous_advantages_lookup,
                    positive_threshold=positive_threshold,
                    max_continuous_advantage=max_continuous_advantage,
                    filter_positive_continuous=True,
                )
                awbc_total_samples += len(transformed_dataset)
                awbc_kept_samples += len(final_dataset)
                if self._rank == 0:
                    self.log_info(
                        f"Loaded AWBC dataset: {data_path} "
                        f"({len(final_dataset)}/{len(transformed_dataset)} samples, "
                        f"weight={weight}, threshold={positive_threshold:.6f}, "
                        f"max_advantage={max_continuous_advantage:.6f}, "
                        f"kept_ratio={final_dataset.kept_ratio:.4f})"
                    )
            elif positive_only:
                advantages_lookup = load_advantages_lookup(data_path, advantage_tag)
                final_dataset = PositiveAdvantageOnlySubset(
                    base_dataset=base_dataset,
                    transformed_dataset=transformed_dataset,
                    advantages_lookup=advantages_lookup,
                )
                if self._rank == 0:
                    adv_filename = (
                        f"advantages_{advantage_tag}.parquet"
                        if advantage_tag
                        else "advantages.parquet"
                    )
                    self.log_info(
                        f"Loaded positive-only dataset: {data_path} "
                        f"({len(final_dataset)}/{len(transformed_dataset)} samples, "
                        f"weight={weight}, meta/{adv_filename})"
                    )
            elif self._rank == 0:
                self.log_info(
                    f"Loaded dataset: {data_path} "
                    f"({len(final_dataset)} samples, weight={weight})"
                )

            datasets_with_weights.append((final_dataset, weight))

        self.awbc_enabled = awbc_enabled
        self.awbc_kept_ratio = (
            awbc_kept_samples / awbc_total_samples
            if awbc_enabled and awbc_total_samples > 0
            else 1.0
        )

        combined_dataset = CfgMixtureDataset(
            datasets=datasets_with_weights,
            mode="train",
            balance_dataset_weights=data_cfg.get("balance_dataset_weights", True),
            seed=data_cfg.get("seed", 42),
        )

        data_num_workers = int(data_cfg.get("num_workers", config.num_workers))
        torch_data_loader = create_distributed_torch_dataloader(
            combined_dataset,
            batch_size=config.batch_size,
            num_workers=data_num_workers,
            world_size=self._world_size,
            rank=self._rank,
            shuffle=True,
        )

        data_loader = SftPlainDataLoaderImpl(data_config, torch_data_loader)
        return data_loader, data_loader.data_config()

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if self._is_lingbotvla_model(model_type):
            batch = _pytree.tree_map(
                lambda x: torch.as_tensor(x, device=self.device).contiguous().clone()
                if isinstance(x, torch.Tensor)
                else x,
                batch,
            )
            with self.amp_context:
                losses_dict = self.model(forward_type=ForwardType.SFT, data=batch)
            return losses_dict["loss"]
        observation, actions = batch[:2]
        advantage_continuous = None
        advantage_weight = None
        if len(batch) >= 4:
            advantage_continuous = batch[2]
            advantage_weight = batch[3]

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: torch.as_tensor(x, device=self.device).contiguous().clone()
            if x is not None
            else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)
        model_data = {"observation": observation, "actions": actions}
        if advantage_continuous is not None and advantage_weight is not None:
            model_data["advantage_continuous"] = advantage_continuous.to(
                self.device, non_blocking=True
            )
            model_data["advantage_weight"] = advantage_weight.to(
                self.device, non_blocking=True
            )

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data=model_data,
            )

        # train model return the loss
        return losses
