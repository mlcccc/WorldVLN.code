#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open


TOP_LEVEL_META_KEYS = ("arch", "epoch", "iter", "acc_str", "g_it")
TRAINER_GROUPS = {
    "gpt": "trainer.gpt_fsdp",
    "vae": "trainer.vae_local",
}
MANIFEST_NAME = "export_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a WorldVLN training checkpoint into a Hugging Face friendly sharded repo layout."
    )
    parser.add_argument("input_path", help="Path to the original WorldVLN .pth checkpoint")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Target repository directory to generate. Upload this directory to Hugging Face.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Maximum shard size for each exported safetensors file, e.g. 10GB",
    )
    parser.add_argument(
        "--model-name",
        default="WorldVLN Backbone",
        help="Display name written into the generated README.md",
    )
    parser.add_argument(
        "--license",
        default="other",
        help="Hugging Face model card license field, e.g. mit / apache-2.0 / other",
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help="Optional Hugging Face repo id to mention in the generated README, e.g. org/name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-export validation",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")
    return checkpoint


def nested_get(obj: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Missing key path: {dotted_path}")
        current = current[part]
    return current


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


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            existing = list(output_dir.iterdir())
            if existing:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {output_dir}. "
                    "Pass --overwrite to replace it."
                )
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def export_group(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: Path,
    max_shard_size: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_state = dict(state_dict)
    save_torch_state_dict(
        export_state,
        save_directory=output_dir,
        filename_pattern="model{suffix}.safetensors",
        force_contiguous=True,
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )

    files = sorted(p.name for p in output_dir.iterdir())
    shard_files = [name for name in files if name.endswith(".safetensors")]
    index_files = [name for name in files if name.endswith(".safetensors.index.json")]
    return {
        "tensor_count": len(state_dict),
        "shard_files": shard_files,
        "index_file": index_files[0] if index_files else "",
    }


def extract_alias_map_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    if not isinstance(metadata, Mapping):
        return alias_map
    for key, value in metadata.items():
        if key == "format":
            continue
        if isinstance(key, str) and isinstance(value, str):
            alias_map[key] = value
    return alias_map


def iter_exported_tensors(export_dir: Path):
    for file_path in sorted(export_dir.glob("*.safetensors")):
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def collect_export_metadata(export_dir: Path) -> tuple[set[str], dict[str, str]]:
    exported_keys: set[str] = set()
    alias_map: dict[str, str] = {}

    index_files = sorted(export_dir.glob("*.safetensors.index.json"))
    if index_files:
        index_data = json.loads(index_files[0].read_text())
        exported_keys.update(index_data.get("weight_map", {}).keys())
        alias_map.update(extract_alias_map_from_metadata(index_data.get("metadata")))

    for file_path in sorted(export_dir.glob("*.safetensors")):
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            exported_keys.update(handle.keys())
            alias_map.update(extract_alias_map_from_metadata(handle.metadata()))

    return exported_keys, alias_map


def validate_group_export(name: str, original_state: Mapping[str, torch.Tensor], export_dir: Path) -> None:
    print(f"[validate] checking {name}: {export_dir}")
    exported_keys, alias_map = collect_export_metadata(export_dir)
    expected_keys = set(original_state.keys())
    semantic_keys = set(exported_keys)

    for alias_key, canonical_key in alias_map.items():
        if canonical_key in expected_keys:
            semantic_keys.add(alias_key)

    missing = sorted(expected_keys - semantic_keys)
    extra = sorted(semantic_keys - expected_keys)
    if missing or extra:
        raise RuntimeError(
            f"{name} exported keys mismatch: missing={missing[:10]} extra={extra[:10]}"
        )

    for key, exported_tensor in iter_exported_tensors(export_dir):
        original_tensor = original_state[key]
        if exported_tensor.shape != original_tensor.shape:
            raise RuntimeError(
                f"{name}:{key} shape mismatch: exported={tuple(exported_tensor.shape)} "
                f"original={tuple(original_tensor.shape)}"
            )
        if exported_tensor.dtype != original_tensor.dtype:
            raise RuntimeError(
                f"{name}:{key} dtype mismatch: exported={exported_tensor.dtype} original={original_tensor.dtype}"
            )
        if not torch.equal(exported_tensor, original_tensor):
            raise RuntimeError(f"{name}:{key} tensor content mismatch after export")

    for alias_key, canonical_key in alias_map.items():
        if alias_key in original_state and canonical_key in original_state:
            if not torch.equal(original_state[alias_key], original_state[canonical_key]):
                raise RuntimeError(
                    f"{name}: alias mapping {alias_key} -> {canonical_key} is not equal in original state dict"
                )


def build_manifest(
    checkpoint: Mapping[str, Any],
    input_path: Path,
    output_dir: Path,
    model_name: str,
    license_name: str,
    repo_id: str,
    max_shard_size: str,
    exports: dict[str, Any],
) -> dict[str, Any]:
    trainer = checkpoint.get("trainer", {})
    top_level_metadata = {key: to_jsonable(checkpoint.get(key)) for key in TOP_LEVEL_META_KEYS}
    top_level_metadata["args"] = to_jsonable(checkpoint.get("args"))
    trainer_metadata = {"config": to_jsonable(trainer.get("config"))}

    return {
        "format": "worldvln_hf_repo_v1",
        "model_name": model_name,
        "license": license_name,
        "repo_id": repo_id,
        "source_checkpoint": str(input_path),
        "output_dir": str(output_dir),
        "max_shard_size": max_shard_size,
        "top_level_metadata": top_level_metadata,
        "trainer_metadata": trainer_metadata,
        "exports": exports,
    }


def build_readme(manifest: Mapping[str, Any]) -> str:
    model_name = manifest["model_name"]
    license_name = manifest["license"]
    repo_id = manifest.get("repo_id", "")
    source_checkpoint = manifest["source_checkpoint"]
    exports = manifest["exports"]
    arch = manifest["top_level_metadata"].get("arch", "")
    epoch = manifest["top_level_metadata"].get("epoch", "")
    iter_idx = manifest["top_level_metadata"].get("iter", "")
    g_it = manifest["top_level_metadata"].get("g_it", "")
    repo_line = f"- Recommended Hugging Face repo id: `{repo_id}`\n" if repo_id else ""

    return textwrap.dedent(
        f"""\
        ---
        license: {license_name}
        library_name: pytorch
        tags:
        - custom-code
        - visual-navigation
        - worldvln
        - safetensors
        ---

        # {model_name}

        This repository was exported from a WorldVLN training checkpoint into a Hugging Face friendly layout.
        It is meant for direct folder upload: upload this whole directory as the root of a Hugging Face model repo.

        ## Included Weights

        - `gpt/`: standard sharded `safetensors` export of `trainer.gpt_fsdp`
        - `vae/`: standard sharded `safetensors` export of `trainer.vae_local`
        - `load_weights.py`: helper utilities for loading the two subfolders directly
        - `{MANIFEST_NAME}`: export provenance and metadata

        ## Source Checkpoint

        - Original checkpoint: `{source_checkpoint}`
        - Architecture: `{arch}`
        - Epoch: `{epoch}`
        - Iter: `{iter_idx}`
        - Global step: `{g_it}`
        {repo_line}
        ## File Layout

        - `gpt/model.safetensors.index.json`
        - `gpt/model-00001-of-xxxxx.safetensors`
        - `vae/model.safetensors.index.json`
        - `vae/model-00001-of-xxxxx.safetensors`

        GPT shard count: `{len(exports["gpt"]["shard_files"])}`

        VAE shard count: `{len(exports["vae"]["shard_files"])}`

        ## Direct Loading

        This export is intentionally split into two model folders instead of one mixed training checkpoint.
        Instantiate your GPT model and VAE model with this project's code, then load them separately.

        ```python
        from load_weights import load_worldvln_models

        load_worldvln_models(
            repo_dir=".",
            gpt_model=infinity_model,
            vae_model=vae_model,
            strict=False,
            device="cpu",
        )
        ```

        Or load raw state dicts only:

        ```python
        from load_weights import load_worldvln_state_dicts

        bundle = load_worldvln_state_dicts(".", device="cpu")
        gpt_state_dict = bundle["gpt"]
        vae_state_dict = bundle["vae"]
        ```

        ## Notes

        - This is a custom-code model export, not a generic `transformers.AutoModel.from_pretrained(...)` repo.
        - The weights are in standard sharded `safetensors` format and do not require manual file concatenation.
        - For inference in this codebase, point the GPT loader to `gpt/` and the VAE loader to `vae/`.
        """
    )


def build_helper_py() -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import Any

        import torch
        from safetensors import safe_open


        MANIFEST_NAME = "{MANIFEST_NAME}"


        def _alias_map_from_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
            alias_map: dict[str, str] = {{}}
            if not isinstance(metadata, dict):
                return alias_map
            for key, value in metadata.items():
                if key == "format":
                    continue
                if isinstance(key, str) and isinstance(value, str):
                    alias_map[key] = value
            return alias_map


        def load_sharded_state_dict(folder: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
            folder = Path(folder)
            target_device = str(device)
            state_dict: dict[str, torch.Tensor] = {{}}
            alias_map: dict[str, str] = {{}}

            index_files = sorted(folder.glob("*.safetensors.index.json"))
            if index_files:
                index_data = json.loads(index_files[0].read_text())
                shard_names = list(dict.fromkeys(index_data["weight_map"].values()))
                alias_map.update(_alias_map_from_metadata(index_data.get("metadata")))
                for shard_name in shard_names:
                    shard_path = folder / shard_name
                    with safe_open(str(shard_path), framework="pt", device=target_device) as handle:
                        alias_map.update(_alias_map_from_metadata(handle.metadata()))
                        for key in handle.keys():
                            state_dict[key] = handle.get_tensor(key)
            else:
                safetensor_files = sorted(folder.glob("*.safetensors"))
                if len(safetensor_files) != 1:
                    raise FileNotFoundError(
                        f"Expected one standalone .safetensors file or one index json in {{folder}}, found {{len(safetensor_files)}}"
                    )
                with safe_open(str(safetensor_files[0]), framework="pt", device=target_device) as handle:
                    alias_map.update(_alias_map_from_metadata(handle.metadata()))
                    for key in handle.keys():
                        state_dict[key] = handle.get_tensor(key)

            for alias_key, canonical_key in alias_map.items():
                if alias_key not in state_dict and canonical_key in state_dict:
                    state_dict[alias_key] = state_dict[canonical_key]
            return state_dict


        def load_worldvln_state_dicts(repo_dir: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
            repo_dir = Path(repo_dir)
            manifest_path = repo_dir / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {{}}
            return {{
                "manifest": manifest,
                "gpt": load_sharded_state_dict(repo_dir / "gpt", device=device),
                "vae": load_sharded_state_dict(repo_dir / "vae", device=device),
            }}


        def load_worldvln_models(
            repo_dir: str | Path,
            *,
            gpt_model: torch.nn.Module | None = None,
            vae_model: torch.nn.Module | None = None,
            strict: bool = False,
            device: str | torch.device = "cpu",
        ) -> dict[str, Any]:
            bundle = load_worldvln_state_dicts(repo_dir, device=device)

            if gpt_model is not None:
                gpt_result = gpt_model.load_state_dict(bundle["gpt"], strict=strict)
            else:
                gpt_result = None

            if vae_model is not None:
                vae_result = vae_model.load_state_dict(bundle["vae"], strict=strict)
            else:
                vae_result = None

            bundle["gpt_load_result"] = gpt_result
            bundle["vae_load_result"] = vae_result
            return bundle
        """
    )


def write_support_files(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "README.md").write_text(build_readme(manifest), encoding="utf-8")
    (output_dir / "load_weights.py").write_text(build_helper_py(), encoding="utf-8")
    (output_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
        "*.pt filter=lfs diff=lfs merge=lfs -text\n"
        "*.pth filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    print(f"[1/6] Loading checkpoint: {input_path}")
    checkpoint = load_checkpoint(input_path)

    trainer = checkpoint.get("trainer")
    if not isinstance(trainer, Mapping):
        raise KeyError("Checkpoint does not contain a 'trainer' mapping")

    exports: dict[str, Any] = {}
    original_states: dict[str, Mapping[str, torch.Tensor]] = {}

    print("[2/6] Collecting export groups")
    for name, dotted_path in TRAINER_GROUPS.items():
        state_dict = nested_get(checkpoint, dotted_path)
        if not is_tensor_state_dict(state_dict):
            raise ValueError(f"{dotted_path} is not a tensor-only state_dict")
        original_states[name] = dict(state_dict)

    print(f"[3/6] Exporting GPT shards -> {output_dir / 'gpt'}")
    exports["gpt"] = export_group(original_states["gpt"], output_dir / "gpt", args.max_shard_size)
    exports["gpt"]["source_key"] = TRAINER_GROUPS["gpt"]

    print(f"[4/6] Exporting VAE shards -> {output_dir / 'vae'}")
    exports["vae"] = export_group(original_states["vae"], output_dir / "vae", args.max_shard_size)
    exports["vae"]["source_key"] = TRAINER_GROUPS["vae"]

    manifest = build_manifest(
        checkpoint=checkpoint,
        input_path=input_path,
        output_dir=output_dir,
        model_name=args.model_name,
        license_name=args.license,
        repo_id=args.repo_id,
        max_shard_size=args.max_shard_size,
        exports=exports,
    )

    print("[5/6] Writing repository support files")
    write_support_files(output_dir, manifest)

    if args.skip_validation:
        print("[6/6] Validation skipped")
    else:
        print("[6/6] Validating exports")
        validate_group_export("gpt", original_states["gpt"], output_dir / "gpt")
        validate_group_export("vae", original_states["vae"], output_dir / "vae")
        print("[validate] all checks passed")

    print("\nExport complete.")
    print(f"HF repo folder: {output_dir}")
    print(f"GPT folder: {output_dir / 'gpt'}")
    print(f"VAE folder: {output_dir / 'vae'}")
    print(f"Model card: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
