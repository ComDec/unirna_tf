#!/usr/bin/env python
"""Verify HuggingFace-uploaded UniRNA models produce identical embeddings to local weights."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

import unirna_tf  # noqa: F401, E402

MODELS = {
    "UniRNA-L8": "unirna_L8",
    "UniRNA-L12": "unirna_L12",
    "UniRNA-L16": "unirna_L16",
}

TEST_SEQUENCES = [
    "AUGCAUGCAUGC",
    "UUUAAACCCGGG",
    "A" * 100,
    "GCGCGCGCGCGCGCGC",
    "AUGCUAGCUAGCUAGCUAGCUAGCUAGCUAGC",
    "NNNNAUGCNNNN",
    "UACGUACGUACGUACGUACG",
    "G" * 200,
    "AUGC" * 128,
    "AUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCGAUCG" * 5,
]


def verify_model(weights_dir: str, model_subdir: str, hf_repo: str, device: str = "cpu"):
    """Compare local model outputs with HuggingFace-hosted model outputs."""
    local_path = os.path.join(weights_dir, model_subdir)
    print(f"\n{'='*60}")
    print(f"Verifying: local={local_path} vs hf={hf_repo}")
    print(f"{'='*60}")

    # Load local model + tokenizer
    local_model = AutoModel.from_pretrained(local_path, trust_remote_code=True).to(device).eval()
    local_tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    # Load HuggingFace model + tokenizer
    hf_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True).to(device).eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)

    all_passed = True
    for i, seq in enumerate(TEST_SEQUENCES):
        local_inputs = local_tokenizer(seq, return_tensors="pt").to(device)
        hf_inputs = hf_tokenizer(seq, return_tensors="pt").to(device)

        assert torch.equal(local_inputs["input_ids"], hf_inputs["input_ids"]), f"Seq {i}: tokenization mismatch"

        with torch.no_grad():
            local_out = local_model(**local_inputs)
            hf_out = hf_model(**hf_inputs)

        lhs_match = torch.equal(local_out.last_hidden_state, hf_out.last_hidden_state)
        pool_match = torch.equal(local_out.pooler_output, hf_out.pooler_output)

        if not (lhs_match and pool_match):
            all_passed = False
            lhs_diff = (local_out.last_hidden_state - hf_out.last_hidden_state).abs().max().item()
            pool_diff = (local_out.pooler_output - hf_out.pooler_output).abs().max().item()
            print(f"  [FAIL] seq {i} (len={len(seq)}): lhs_maxdiff={lhs_diff:.2e}, pool_maxdiff={pool_diff:.2e}")
        else:
            print(f"  [PASS] seq {i} (len={len(seq)}): bit-exact")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify HuggingFace UniRNA models match local weights")
    parser.add_argument("--weights-dir", required=True, help="Path to local weights directory")
    parser.add_argument("--org", default="EscheWang", help="HuggingFace org/user")
    parser.add_argument("--models", nargs="*", default=list(MODELS.keys()), help="Models to verify")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    args = parser.parse_args()

    results = {}
    for hf_name in args.models:
        if hf_name not in MODELS:
            print(f"Unknown model: {hf_name}, skipping")
            continue
        passed = verify_model(args.weights_dir, MODELS[hf_name], f"{args.org}/{hf_name}", args.device)
        results[hf_name] = passed

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    if not all(results.values()):
        sys.exit(1)
    print("\nAll models verified successfully!")


if __name__ == "__main__":
    main()
