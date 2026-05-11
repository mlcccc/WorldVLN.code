#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import save_torch_state_dict
from safetensors.torch import load_file as load_safetensors_file


DEFAULT_GROUPS = ("trainer.vae_local", "trainer.gpt_fsdp")
MANIFEST_NAME = "export_manifest.json"


def is_tensor_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values())


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return repr(obj)


def nested_get(obj: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Missing key path: {dotted_path}")
        current = current[part]
    return current


def nested_set(obj: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = obj
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_manifest(checkpoint: Mapping[str, Any], source_path: Path, groups: list[str]) -> dict[str, Any]:
    trainer = checkpoint.get("trainer", {})
    manifest = {
        "format": "worldvln_sharded_safetensors_v1",
        "source_checkpoint": str(source_path),
        "exported_groups": groups,
        "top_level_metadata": {
            "args": to_jsonable(checkpoint.get("args")),
            "arch": to_jsonable(checkpoint.get("arch")),
            "epoch": to_jsonable(checkpoint.get("epoch")),
            "iter": to_jsonable(checkpoint.get("iter")),
            "acc_str": to_jsonable(checkpoint.get("acc_str")),
            "g_it": to_jsonable(checkpoint.get("g_it")),
        },
        "trainer_metadata": {
            "config": to_jsonable(trainer.get("config")),
        },
    }
    return manifest


def flatten_export_groups(checkpoint: Mapping[str, Any], groups: list[str]) -> OrderedDict[str, torch.Tensor]:
    merged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for group_path in groups:
        state_dict = nested_get(checkpoint, group_path)
        if not is_tensor_state_dict(state_dict):
            raise ValueError(f"{group_path} is not a pure tensor state_dict")
        prefix = f"{group_path}."
        for key, tensor in state_dict.items():
            merged_key = f"{prefix}{key}"
            if merged_key in merged_state:
                raise KeyError(f"Duplicated merged key: {merged_key}")
            merged_state[merged_key] = tensor.detach().contiguous()
    return merged_state


def autodetect_index_file(checkpoint_dir: Path) -> Path:
    matches = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*.safetensors.index.json' in {checkpoint_dir}, found {len(matches)}"
        )
    return matches[0]


def device_string(device: str | torch.device) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return device


def load_flat_sharded_state(checkpoint_dir: Path, device: str | torch.device = "cpu") -> OrderedDict[str, torch.Tensor]:
    merged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    target_device = device_string(device)

    index_files = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if index_files:
        index_path = autodetect_index_file(checkpoint_dir)
        index_data = json.loads(index_path.read_text())
        shard_names = list(dict.fromkeys(index_data["weight_map"].values()))
        for shard_name in shard_names:
            shard_path = checkpoint_dir / shard_name
            shard_state = load_safetensors_file(str(shard_path), device=target_device)
            merged_state.update(shard_state)

        # `save_torch_state_dict` may deduplicate shared tensors and store alias
        # information inside index metadata as: {"alias_key": "canonical_key"}.
        for alias_key, canonical_key in index_data.get("metadata", {}).items():
            if (
                isinstance(alias_key, str)
                and isinstance(canonical_key, str)
                and alias_key not in merged_state
                and canonical_key in merged_state
            ):
                merged_state[alias_key] = merged_state[canonical_key]
        return merged_state

    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if len(safetensor_files) != 1:
        raise FileNotFoundError(
            f"Expected one standalone '*.safetensors' file in {checkpoint_dir}, found {len(safetensor_files)}"
        )
    merged_state.update(load_safetensors_file(str(safetensor_files[0]), device=target_device))
    return merged_state


def load_sharded_worldvln_checkpoint(checkpoint_dir: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    exported_groups = manifest.get("exported_groups", [])
    flat_state = load_flat_sharded_state(checkpoint_dir, device=device)

    checkpoint: dict[str, Any] = dict(manifest.get("top_level_metadata", {}))
    trainer_metadata = manifest.get("trainer_metadata", {})
    if trainer_metadata:
        checkpoint["trainer"] = dict(trainer_metadata)

    for group_path in exported_groups:
        prefix = f"{group_path}."
        group_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, tensor in flat_state.items():
            if key.startswith(prefix):
                group_state[key[len(prefix):]] = tensor
        if not group_state:
            raise KeyError(f"No tensors found for group {group_path}")
        nested_set(checkpoint, group_path, group_state)

    return checkpoint


def export_checkpoint(
    input_path: Path,
    output_dir: Path,
    prefix: str,
    max_shard_size: str,
    groups: list[str],
) -> None:
    print(f"[1/4] Loading checkpoint: {input_path}")
    checkpoint = load_torch_checkpoint(input_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

    print(f"[2/4] Collecting groups: {', '.join(groups)}")
    merged_state = flatten_export_groups(checkpoint, groups)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Saving safetensors shards to: {output_dir}")
    save_torch_state_dict(
        merged_state,
        save_directory=output_dir,
        filename_pattern=f"{prefix}{{suffix}}.safetensors",
        force_contiguous=True,
        max_shard_size=max_shard_size,
        metadata={"source_checkpoint": input_path.name},
        safe_serialization=True,
    )

    manifest = build_manifest(checkpoint, input_path, groups)
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print("[4/4] Done.")
    print(f"        manifest: {output_dir / MANIFEST_NAME}")
    for path in sorted(output_dir.glob(f"{prefix}*.safetensors*")):
        print(f"        file: {path.name}")


def inspect_checkpoint(input_path: Path) -> None:
    checkpoint = load_torch_checkpoint(input_path)
    print(f"type={type(checkpoint)}")
    if not isinstance(checkpoint, Mapping):
        return

    print(f"top_level_keys={list(checkpoint.keys())}")
    trainer = checkpoint.get("trainer")
    if isinstance(trainer, Mapping):
        print(f"trainer_keys={list(trainer.keys())}")
        for key, value in trainer.items():
            if is_tensor_state_dict(value):
                print(f"trainer.{key}: tensor_state_dict with {len(value)} tensors")
            else:
                print(f"trainer.{key}: {type(value)}")


def cmd_export(args: argparse.Namespace) -> None:
    export_checkpoint(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        max_shard_size=args.max_shard_size,
        groups=list(args.groups),
    )


def cmd_load(args: argparse.Namespace) -> None:
    checkpoint = load_sharded_worldvln_checkpoint(args.checkpoint_dir, device=args.device)

    print("Loaded sharded checkpoint.")
    print(f"top_level_keys={list(checkpoint.keys())}")
    trainer = checkpoint.get("trainer", {})
    if isinstance(trainer, Mapping):
        for key, value in trainer.items():
            if is_tensor_state_dict(value):
                print(f"trainer.{key}: tensor_state_dict with {len(value)} tensors")
            else:
                print(f"trainer.{key}: {type(value)}")

    if args.save_pth:
        save_path = Path(args.save_pth)
        torch.save(checkpoint, save_path)
        print(f"Repacked checkpoint saved to: {save_path}")


def cmd_inspect(args: argparse.Namespace) -> None:
    inspect_checkpoint(Path(args.input_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a WorldVLN-style .pth checkpoint into sharded safetensors and load it back."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect checkpoint structure")
    inspect_parser.add_argument("input_path", help="Path to the original .pth checkpoint")
    inspect_parser.set_defaults(func=cmd_inspect)

    export_parser = subparsers.add_parser("export", help="Export selected weight groups into safetensors shards")
    export_parser.add_argument("input_path", help="Path to the original .pth checkpoint")
    export_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write safetensors shards, index json, and manifest",
    )
    export_parser.add_argument("--prefix", default="WorldVLN_backbone", help="Shard filename prefix")
    export_parser.add_argument("--max-shard-size", default="10GB", help="Max size per shard, e.g. 10GB")
    export_parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Dotted key paths to tensor state_dict groups inside the checkpoint",
    )
    export_parser.set_defaults(func=cmd_export)

    load_parser = subparsers.add_parser("load", help="Load a sharded export back into one checkpoint dict")
    load_parser.add_argument("checkpoint_dir", help="Directory that contains sharded safetensors export")
    load_parser.add_argument("--device", default="cpu", help="Device to load tensors onto")
    load_parser.add_argument("--save-pth", default="", help="Optional path to repack the loaded checkpoint as .pth")
    load_parser.set_defaults(func=cmd_load)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
