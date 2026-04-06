#!/usr/bin/env python
"""Upload UniRNA models to HuggingFace Hub with custom remote code."""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from huggingface_hub import HfApi  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

import unirna_tf  # noqa: F401, E402 — triggers Auto* registration

MODELS = {
    "UniRNA-L8": "unirna_L8",
    "UniRNA-L12": "unirna_L12",
    "UniRNA-L16": "unirna_L16",
}

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "unirna_tf")
CODE_FILES = ["config.py", "model.py", "tokenizer.py"]

AUTO_MAP_CONFIG = {
    "AutoConfig": "config.UniRNAConfig",
    "AutoModel": "model.UniRNAModels",
    "AutoModelForMaskedLM": "model.UniRNAForMaskedLM",
}

AUTO_MAP_TOKENIZER = {
    "AutoTokenizer": ["tokenizer.UniRNATokenizer", None],
}


def prepare_staging(weights_dir: str, staging_dir: str, model_subdir: str, hf_name: str):
    """Prepare a staging directory for one model."""
    src = os.path.join(weights_dir, model_subdir)
    dst = os.path.join(staging_dir, hf_name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    # 1. Load model and convert to safetensors
    print(f"[{hf_name}] Loading model from {src} ...")
    mlm_model = AutoModelForMaskedLM.from_pretrained(src, trust_remote_code=True)
    mlm_model.eval()

    state_dict = {k: v.contiguous() for k, v in mlm_model.state_dict().items()}
    safetensors_path = os.path.join(dst, "model.safetensors")
    save_file(state_dict, safetensors_path)
    print(f"[{hf_name}] Saved safetensors ({os.path.getsize(safetensors_path) / 1e6:.1f} MB)")

    # 2. Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    tokenizer.save_pretrained(dst)

    # 3. Write config.json with auto_map
    with open(os.path.join(src, "config.json")) as f:
        config_data = json.load(f)
    config_data["auto_map"] = AUTO_MAP_CONFIG
    config_data.pop("_name_or_path", None)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config_data, f, indent=2)
        f.write("\n")
    print(f"[{hf_name}] Wrote config.json with auto_map")

    # 4. Write tokenizer_config.json with auto_map
    tok_config_path = os.path.join(dst, "tokenizer_config.json")
    with open(tok_config_path) as f:
        tok_data = json.load(f)
    tok_data["auto_map"] = AUTO_MAP_TOKENIZER
    with open(tok_config_path, "w") as f:
        json.dump(tok_data, f, indent=2)
        f.write("\n")
    print(f"[{hf_name}] Wrote tokenizer_config.json with auto_map")

    # 5. Copy Python source files for remote code
    for fname in CODE_FILES:
        shutil.copy2(os.path.join(SOURCE_DIR, fname), os.path.join(dst, fname))
    print(f"[{hf_name}] Copied {len(CODE_FILES)} source files")

    return dst


def upload(staging_path: str, repo_id: str):
    """Upload staging directory to HuggingFace Hub."""
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=staging_path, repo_id=repo_id)
    print(f"Uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload UniRNA models to HuggingFace Hub")
    parser.add_argument("--weights-dir", required=True, help="Path to weights directory")
    parser.add_argument("--staging-dir", default="/tmp/unirna_hf_staging", help="Staging directory")
    parser.add_argument("--org", default="EscheWang", help="HuggingFace org/user")
    parser.add_argument("--models", nargs="*", default=list(MODELS.keys()), help="Models to upload")
    parser.add_argument("--skip-upload", action="store_true", help="Only prepare staging, skip upload")
    args = parser.parse_args()

    for hf_name in args.models:
        if hf_name not in MODELS:
            print(f"Unknown model: {hf_name}, skipping")
            continue
        staging_path = prepare_staging(args.weights_dir, args.staging_dir, MODELS[hf_name], hf_name)
        if not args.skip_upload:
            upload(staging_path, f"{args.org}/{hf_name}")
        else:
            print(f"[{hf_name}] Staging ready at {staging_path} (upload skipped)")


if __name__ == "__main__":
    main()
