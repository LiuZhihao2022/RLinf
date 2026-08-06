#!/usr/bin/env bash
set -euo pipefail

# Patch the currently active Python environment's installed openpi package so it
# supports state history/future windows used by LeRobotX2robotDataConfig.
#
# Usage:
#   source switch_env openpi
#   bash toolkits/patch_openpi_state_sequence.sh

python - <<'PY'
from __future__ import annotations

import dataclasses
import pathlib
import py_compile

import openpi


OPENPI_DIR = pathlib.Path(openpi.__file__).resolve().parent


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def write_if_changed(path: pathlib.Path, text: str, old_text: str) -> None:
    if text == old_text:
        print(f"unchanged: {path}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"patched:   {path}")


def replace_once(text: str, old: str, new: str, path: pathlib.Path, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find patch target '{label}' in {path}")
    return text.replace(old, new, 1)


def patch_config() -> pathlib.Path:
    path = OPENPI_DIR / "training" / "config.py"
    original = read(path)
    text = original

    if "state_history_size: int = 0" not in text:
        old = (
            "    # If true, will use the LeRobot dataset task to define the prompt.\n"
            "    prompt_from_task: bool = False\n"
        )
        new = old + "    state_history_size: int = 0\n    state_future_size: int = 0\n"
        text = replace_once(text, old, new, path, "DataConfig state window fields")

    if "_transforms.BuildStateSequence" not in text:
        old = (
            "        match model_config.model_type:\n"
            "            case _model.ModelType.PI0:\n"
            "                return _transforms.Group(\n"
            "                    inputs=[\n"
            "                        _transforms.InjectDefaultPrompt(self.default_prompt),\n"
            "                        _transforms.ResizeImages(224, 224),\n"
            "                        _transforms.TokenizePrompt(\n"
            "                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),\n"
            "                        ),\n"
            "                        _transforms.PadStatesAndActions(model_config.action_dim),\n"
            "                    ],\n"
            "                )\n"
            "            case _model.ModelType.PI05:\n"
            "                assert isinstance(model_config, pi0_config.Pi0Config)\n"
            "                return _transforms.Group(\n"
            "                    inputs=[\n"
            "                        _transforms.InjectDefaultPrompt(self.default_prompt),\n"
            "                        _transforms.ResizeImages(224, 224),\n"
            "                        _transforms.TokenizePrompt(\n"
            "                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),\n"
            "                            discrete_state_input=model_config.discrete_state_input,\n"
            "                        ),\n"
            "                        _transforms.PadStatesAndActions(model_config.action_dim),\n"
            "                    ],\n"
            "                )\n"
        )
        new = (
            "        match model_config.model_type:\n"
            "            case _model.ModelType.PI0 | _model.ModelType.PI05:\n"
            "                assert isinstance(model_config, pi0_config.Pi0Config)\n"
            "                discrete_state_input = model_config.discrete_state_input if model_config.pi05 else False\n"
            "                input_transforms = [\n"
            "                    _transforms.InjectDefaultPrompt(self.default_prompt),\n"
            "                    _transforms.ResizeImages(224, 224),\n"
            "                    _transforms.TokenizePrompt(\n"
            "                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),\n"
            "                        discrete_state_input=discrete_state_input,\n"
            "                    ),\n"
            "                ]\n"
            "                if model_config.state_sequence_length > 1:\n"
            "                    input_transforms.append(_transforms.BuildStateSequence(model_config.state_sequence_length))\n"
            "                input_transforms.append(_transforms.PadStatesAndActions(model_config.action_dim))\n"
            "                return _transforms.Group(inputs=input_transforms)\n"
        )
        text = replace_once(text, old, new, path, "ModelTransformFactory state sequence")

    if "Auto-sync state_sequence_length" not in text:
        old = (
            "    def __post_init__(self) -> None:\n"
            "        if self.resume and self.overwrite:\n"
            "            raise ValueError(\"Cannot resume and overwrite at the same time.\")\n"
        )
        new = (
            "    def __post_init__(self) -> None:\n"
            "        if self.resume and self.overwrite:\n"
            "            raise ValueError(\"Cannot resume and overwrite at the same time.\")\n"
            "\n"
            "        # Auto-sync state_sequence_length from data to model for datasets that\n"
            "        # request state history/future windows.\n"
            "        data_seq_len = getattr(self.data, \"state_sequence_length\", 1)\n"
            "        model_seq_len = getattr(self.model, \"state_sequence_length\", 1)\n"
            "\n"
            "        if data_seq_len > 1 and model_seq_len == 1:\n"
            "            new_model = dataclasses.replace(self.model, state_sequence_length=data_seq_len)\n"
            "            object.__setattr__(self, \"model\", new_model)\n"
            "        elif model_seq_len != data_seq_len and model_seq_len != 1:\n"
            "            raise ValueError(\n"
            "                f\"Mismatch: model.state_sequence_length={model_seq_len}, \"\n"
            "                f\"data.state_sequence_length={data_seq_len}\"\n"
            "            )\n"
        )
        text = replace_once(text, old, new, path, "TrainConfig state_sequence_length sync")

    write_if_changed(path, text, original)
    return path


def patch_data_loader() -> pathlib.Path:
    path = OPENPI_DIR / "training" / "data_loader.py"
    original = read(path)
    text = original

    if 'state_history_size = getattr(data_config, "state_history_size", 0)' not in text:
        old = (
            "    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)\n"
            "    dataset = lerobot_dataset.LeRobotDataset(\n"
            "        data_config.repo_id,\n"
            "        delta_timestamps={\n"
            "            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys\n"
            "        },\n"
            "    )\n"
        )
        new = (
            "    state_history_size = getattr(data_config, \"state_history_size\", 0)\n"
            "    state_future_size = getattr(data_config, \"state_future_size\", 0)\n"
            "    state_step = getattr(data_config, \"state_step\", 1)\n"
            "\n"
            "    def _build_delta_timestamps(fps: float) -> dict[str, list[float]]:\n"
            "        delta_ts = {\n"
            "            key: [t / fps for t in range(action_horizon)]\n"
            "            for key in data_config.action_sequence_keys\n"
            "        }\n"
            "        if state_history_size > 0 or state_future_size > 0:\n"
            "            delta_ts[\"state\"] = [\n"
            "                t * state_step / fps\n"
            "                for t in range(-state_history_size, state_future_size + 1)\n"
            "            ]\n"
            "        return delta_ts\n"
            "\n"
            "    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)\n"
            "    dataset = lerobot_dataset.LeRobotDataset(\n"
            "        data_config.repo_id,\n"
            "        delta_timestamps=_build_delta_timestamps(dataset_meta.fps),\n"
            "    )\n"
        )
        text = replace_once(text, old, new, path, "LeRobot state delta_timestamps")

    # Multi-dataset support: split comma-separated repo_id, build per-dataset,
    # then ConcatDataset.
    if "repo_ids = [r.strip()" not in text:
        old = (
            "    state_history_size = getattr(data_config, \"state_history_size\", 0)\n"
            "    state_future_size = getattr(data_config, \"state_future_size\", 0)\n"
            "    state_step = getattr(data_config, \"state_step\", 1)\n"
            "\n"
            "    def _build_delta_timestamps(fps: float) -> dict[str, list[float]]:\n"
            "        delta_ts = {\n"
            "            key: [t / fps for t in range(action_horizon)]\n"
            "            for key in data_config.action_sequence_keys\n"
            "        }\n"
            "        if state_history_size > 0 or state_future_size > 0:\n"
            "            delta_ts[\"state\"] = [\n"
            "                t * state_step / fps\n"
            "                for t in range(-state_history_size, state_future_size + 1)\n"
            "            ]\n"
            "        return delta_ts\n"
            "\n"
            "    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)\n"
            "    dataset = lerobot_dataset.LeRobotDataset(\n"
            "        data_config.repo_id,\n"
            "        delta_timestamps=_build_delta_timestamps(dataset_meta.fps),\n"
            "    )\n"
            "\n"
            "    if data_config.prompt_from_task:\n"
            "        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])\n"
            "\n"
            "    return dataset\n"
        )
        new = (
            "    state_history_size = getattr(data_config, \"state_history_size\", 0)\n"
            "    state_future_size = getattr(data_config, \"state_future_size\", 0)\n"
            "    state_step = getattr(data_config, \"state_step\", 1)\n"
            "\n"
            "    def _build_delta_timestamps(fps: float) -> dict[str, list[float]]:\n"
            "        delta_ts = {\n"
            "            key: [t / fps for t in range(action_horizon)]\n"
            "            for key in data_config.action_sequence_keys\n"
            "        }\n"
            "        if state_history_size > 0 or state_future_size > 0:\n"
            "            delta_ts[\"state\"] = [\n"
            "                t * state_step / fps\n"
            "                for t in range(-state_history_size, state_future_size + 1)\n"
            "            ]\n"
            "        return delta_ts\n"
            "\n"
            "    def _build_single_dataset(single_repo_id: str):\n"
            "        meta = lerobot_dataset.LeRobotDatasetMetadata(single_repo_id)\n"
            "        ds = lerobot_dataset.LeRobotDataset(\n"
            "            single_repo_id,\n"
            "            delta_timestamps=_build_delta_timestamps(meta.fps),\n"
            "        )\n"
            "        if data_config.prompt_from_task:\n"
            "            ds = TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(meta.tasks)])\n"
            "        return ds\n"
            "\n"
            "    repo_ids = [r.strip() for r in repo_id.split(\",\") if r.strip()]\n"
            "    if len(repo_ids) == 1:\n"
            "        return _build_single_dataset(repo_ids[0])\n"
            "\n"
            "    from torch.utils.data import ConcatDataset as _ConcatDataset\n"
            "    return _ConcatDataset([_build_single_dataset(rid) for rid in repo_ids])\n"
        )
        text = replace_once(text, old, new, path, "Multi-dataset ConcatDataset support")

    write_if_changed(path, text, original)
    return path


def patch_pi0_config() -> pathlib.Path:
    path = OPENPI_DIR / "models" / "pi0_config.py"
    original = read(path)
    text = original

    if "state_sequence_length: int = 1" not in text:
        old = (
            "    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.\n"
            "    discrete_state_input: bool = None  # type: ignore\n"
        )
        new = (
            old
            + "    # Number of state frames: history + current + future. Auto-set by TrainConfig.\n"
            + "    state_sequence_length: int = 1\n"
        )
        text = replace_once(text, old, new, path, "Pi0Config state_sequence_length field")

    if "state_shape = [batch_size, self.state_sequence_length, self.action_dim]" not in text:
        old = (
            "        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)\n"
            "        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)\n"
            "\n"
            "        with at.disable_typechecking():\n"
        )
        new = (
            "        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)\n"
            "        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)\n"
            "\n"
            "        # State shape depends on whether we use state history/future.\n"
            "        if self.state_sequence_length > 1:\n"
            "            state_shape = [batch_size, self.state_sequence_length, self.action_dim]\n"
            "        else:\n"
            "            state_shape = [batch_size, self.action_dim]\n"
            "\n"
            "        with at.disable_typechecking():\n"
        )
        text = replace_once(text, old, new, path, "Pi0Config state_shape")

    old_state_spec = "                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),\n"
    if old_state_spec in text:
        text = text.replace(
            old_state_spec,
            "                state=jax.ShapeDtypeStruct(state_shape, jnp.float32),\n",
            1,
        )

    write_if_changed(path, text, original)
    return path


def patch_transforms() -> pathlib.Path:
    path = OPENPI_DIR / "transforms.py"
    original = read(path)
    text = original

    if "class BuildStateSequence" not in text:
        old = (
            "def flatten_dict(tree: at.PyTree) -> dict:\n"
            "    \"\"\"Flatten a nested dictionary. Uses '/' as the separator.\"\"\"\n"
        )
        new = (
            "@dataclasses.dataclass(frozen=True)\n"
            "class BuildStateSequence(DataTransformFn):\n"
            "    \"\"\"Builds state sequence of shape (sequence_length, state_dim).\"\"\"\n"
            "\n"
            "    state_sequence_length: int = 1\n"
            "    replicate_if_missing: bool = True\n"
            "\n"
            "    def __call__(self, data: DataDict) -> DataDict:\n"
            "        if self.state_sequence_length <= 1:\n"
            "            return data\n"
            "\n"
            "        state = data.get(\"state\")\n"
            "        if state is None:\n"
            "            raise ValueError(\"Missing 'state' in data\")\n"
            "\n"
            "        if state.ndim == 2:\n"
            "            if state.shape[0] >= self.state_sequence_length:\n"
            "                data[\"state\"] = state[: self.state_sequence_length]\n"
            "                return data\n"
            "        elif state.ndim == 1 and self.replicate_if_missing:\n"
            "            data[\"state\"] = np.tile(state[None, :], (self.state_sequence_length, 1))\n"
            "            return data\n"
            "\n"
            "        raise ValueError(\n"
            "            f\"Cannot build state sequence of length {self.state_sequence_length} \"\n"
            "            f\"from state shape {state.shape}\"\n"
            "        )\n"
            "\n\n"
            + old
        )
        text = replace_once(text, old, new, path, "BuildStateSequence transform")

    write_if_changed(path, text, original)
    return path


def patch_pi0() -> pathlib.Path:
    path = OPENPI_DIR / "models" / "pi0.py"
    original = read(path)
    text = original

    if "num_state_tokens = state.shape[1]" not in text:
        old = (
            "        if not self.pi05:\n"
            "            # add a single state token\n"
            "            state_token = self.state_proj(obs.state)[:, None, :]\n"
            "            tokens.append(state_token)\n"
            "            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))\n"
            "            # image/language inputs do not attend to state or actions\n"
            "            ar_mask += [True]\n"
        )
        new = (
            "        if not self.pi05:\n"
            "            # State: (batch, state_dim) or (batch, seq_len, state_dim)\n"
            "            state = obs.state\n"
            "            if state.ndim == 2:\n"
            "                state = state[:, None, :]\n"
            "            num_state_tokens = state.shape[1]\n"
            "\n"
            "            state_tokens = self.state_proj(state)\n"
            "            tokens.append(state_tokens)\n"
            "            input_mask.append(jnp.ones((state.shape[0], num_state_tokens), dtype=jnp.bool_))\n"
            "            ar_mask += [True] + ([False] * (num_state_tokens - 1))\n"
        )
        text = replace_once(text, old, new, path, "Pi0 state token sequence")

    write_if_changed(path, text, original)
    return path


def patch_pi0_pytorch() -> pathlib.Path:
    path = OPENPI_DIR / "models_pytorch" / "pi0_pytorch.py"
    original = read(path)
    text = original

    if "num_state_tokens = state.shape[1]" not in text:
        old = (
            "        if not self.pi05:\n"
            "            if self.state_proj.weight.dtype == torch.float32:\n"
            "                state = state.to(torch.float32)\n"
            "\n"
            "            # Embed state\n"
            "            def state_proj_func(state):\n"
            "                return self.state_proj(state)\n"
            "\n"
            "            state_emb = self._apply_checkpoint(state_proj_func, state)\n"
            "\n"
            "            embs.append(state_emb[:, None, :])\n"
            "            bsize = state_emb.shape[0]\n"
            "            device = state_emb.device\n"
            "\n"
            "            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)\n"
            "            pad_masks.append(state_mask)\n"
            "\n"
            "            # Set attention masks so that image and language inputs do not attend to state or actions\n"
            "            att_masks += [1]\n"
        )
        new = (
            "        if not self.pi05:\n"
            "            if self.state_proj.weight.dtype == torch.float32:\n"
            "                state = state.to(torch.float32)\n"
            "\n"
            "            # State: (batch, state_dim) or (batch, seq_len, state_dim)\n"
            "            if state.ndim == 2:\n"
            "                state = state[:, None, :]\n"
            "            num_state_tokens = state.shape[1]\n"
            "\n"
            "            def state_proj_func(state):\n"
            "                return self.state_proj(state)\n"
            "\n"
            "            state_emb = self._apply_checkpoint(state_proj_func, state)\n"
            "\n"
            "            embs.append(state_emb)\n"
            "            bsize = state_emb.shape[0]\n"
            "            device = state_emb.device\n"
            "\n"
            "            state_mask = torch.ones(\n"
            "                bsize, num_state_tokens, dtype=torch.bool, device=device\n"
            "            )\n"
            "            pad_masks.append(state_mask)\n"
            "\n"
            "            # Match JAX pi0: one new attention block for all state tokens.\n"
            "            att_masks += [1] + ([0] * (num_state_tokens - 1))\n"
        )
        text = replace_once(text, old, new, path, "PI0Pytorch state token sequence")

    write_if_changed(path, text, original)
    return path


def patch_model() -> pathlib.Path:
    path = OPENPI_DIR / "models" / "model.py"
    original = read(path)
    text = original

    old = (
        "    # Low-dimensional robot state.\n"
        "    state: at.Float[ArrayT, \"*b s\"]\n"
    )
    if old in text:
        text = text.replace(
            old,
            "    # Low-dimensional robot state: (batch, state_dim) or (batch, seq_len, state_dim).\n"
            "    state: at.Float[ArrayT, \"...\"]\n",
            1,
        )

    old_batch_shape = "    batch_shape = observation.state.shape[:-1]\n"
    if old_batch_shape in text:
        text = text.replace(old_batch_shape, "    batch_shape = observation.state.shape[:1]\n", 1)

    write_if_changed(path, text, original)
    return path


patched_paths = [
    patch_config(),
    patch_data_loader(),
    patch_pi0_config(),
    patch_transforms(),
    patch_pi0(),
    patch_pi0_pytorch(),
    patch_model(),
]

for path in patched_paths:
    py_compile.compile(str(path), doraise=True)

from openpi.models.pi0_config import Pi0Config
from openpi.training.config import DataConfig
from openpi.transforms import BuildStateSequence

data_fields = {field.name for field in dataclasses.fields(DataConfig)}
model_fields = {field.name for field in dataclasses.fields(Pi0Config)}
assert {"state_history_size", "state_future_size"} <= data_fields
assert "state_sequence_length" in model_fields
assert BuildStateSequence(3)

print(f"openpi patched successfully at: {OPENPI_DIR}")
PY
