import argparse
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from chanfig import NestedDict

from unirna_tf.config import build_config, build_config_GENE


SUPPORTED_TOKENIZER_VERSIONS = {"single", "bpe", "plant_bpe"}


def convert_ckpt(ckpt):
    if isinstance(ckpt, str):
        ckpt = torch.load(ckpt)
    ckpt = NestedDict(ckpt)
    weights = OrderedDict()
    weights["embeddings.word_embeddings.weight"] = ckpt.pop("embed_tokens.weight")
    if "embeddings.position_ids" in ckpt:
        weights["embeddings.position_ids"] = ckpt.pop("embeddings.position_ids")
    if "embeddings.position_embeddings.weight" in ckpt:
        weights["embeddings.position_embeddings.weight"] = ckpt.pop("embeddings.position_embeddings.weight")
    weights["embeddings.layer_norm.weight"] = ckpt.pop("emb_layer_norm_before.weight")
    weights["embeddings.layer_norm.bias"] = ckpt.pop("emb_layer_norm_before.bias")
    for key, value in ckpt.layers.items():
        qw, kw, vw = value.pop("self_attn.in_proj.weight").chunk(3, dim=0)
        qb, kb, vb = value.pop("self_attn.in_proj.bias").chunk(3, dim=0)
        weights[f"encoder.layer.{key}.attention.self.query.weight"] = qw
        weights[f"encoder.layer.{key}.attention.self.query.bias"] = qb
        weights[f"encoder.layer.{key}.attention.self.key.weight"] = kw
        weights[f"encoder.layer.{key}.attention.self.key.bias"] = kb
        weights[f"encoder.layer.{key}.attention.self.value.weight"] = vw
        weights[f"encoder.layer.{key}.attention.self.value.bias"] = vb
        weights[f"encoder.layer.{key}.attention.self.rotary_embeddings.inv_freq"] = value.pop(
            "self_attn.rot_emb.inv_freq"
        )
        weights[f"encoder.layer.{key}.attention.output.dense.weight"] = value.pop("self_attn.out_proj.weight")
        weights[f"encoder.layer.{key}.attention.output.dense.bias"] = value.pop("self_attn.out_proj.bias")
        weights[f"encoder.layer.{key}.attention.LayerNorm.weight"] = value.pop("self_attn_layer_norm.weight")
        weights[f"encoder.layer.{key}.attention.LayerNorm.bias"] = value.pop("self_attn_layer_norm.bias")
        weights[f"encoder.layer.{key}.intermediate.dense.weight"] = value.pop("fc1.weight")
        weights[f"encoder.layer.{key}.intermediate.dense.bias"] = value.pop("fc1.bias")
        weights[f"encoder.layer.{key}.output.dense.weight"] = value.pop("fc2.weight")
        weights[f"encoder.layer.{key}.output.dense.bias"] = value.pop("fc2.bias")
        weights[f"encoder.layer.{key}.LayerNorm.weight"] = value.pop("final_layer_norm.weight")
        weights[f"encoder.layer.{key}.LayerNorm.bias"] = value.pop("final_layer_norm.bias")
    weights["encoder.emb_layer_norm_after.weight"] = ckpt.pop("emb_layer_norm_after.weight")
    weights["encoder.emb_layer_norm_after.bias"] = ckpt.pop("emb_layer_norm_after.bias")
    weights["lm_head.dense.weight"] = ckpt.pop("lm_head.dense.weight")
    weights["lm_head.dense.bias"] = ckpt.pop("lm_head.dense.bias")
    weights["lm_head.layer_norm.weight"] = ckpt.pop("lm_head.layer_norm.weight")
    weights["lm_head.layer_norm.bias"] = ckpt.pop("lm_head.layer_norm.bias")
    weights["lm_head.decoder.weight"] = ckpt.pop("lm_head.out_proj.weight")
    weights["lm_head.decoder.bias"] = ckpt.pop("lm_head.out_proj.bias")
    return weights


def _validate_version(version: str) -> str:
    if version is None:
        return "single"
    normalized = str(version).strip().lower()
    if normalized not in SUPPORTED_TOKENIZER_VERSIONS:
        valid = ", ".join(sorted(SUPPORTED_TOKENIZER_VERSIONS))
        raise ValueError(f"Unsupported tokenizer version '{version}'. Expected one of: {valid}.")
    return normalized


def _validate_gene_dims(num_hidden_layers: int, hidden_size: int, vocab_size: int) -> tuple[int, int, int]:
    try:
        validated = (int(num_hidden_layers), int(hidden_size), int(vocab_size))
    except (TypeError, ValueError) as exc:
        raise ValueError("GENE conversion requires integer values for layers, hidden size, and vocab size.") from exc
    if any(value <= 0 for value in validated):
        raise ValueError("GENE conversion requires positive values for layers, hidden size, and vocab size.")
    return validated


def _resolve_output_dir(path: str) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    return checkpoint_path.with_suffix("")


def _prepare_output_dir(output_dir: Path, tokenizer_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(__file__).resolve().parent / "tokenizer" / tokenizer_name
    shutil.copytree(tokenizer_dir, output_dir, dirs_exist_ok=True)


def _resize_vocab_tensor_rows(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    current_size = tensor.shape[0]
    if current_size == target_size:
        return tensor
    if current_size > target_size:
        return tensor[:target_size].clone()
    new_shape = (target_size, *tensor.shape[1:])
    resized = tensor.new_zeros(new_shape)
    resized[:current_size] = tensor
    return resized


def _resize_vocab_bias(bias: torch.Tensor, target_size: int) -> torch.Tensor:
    current_size = bias.shape[0]
    if current_size == target_size:
        return bias
    if current_size > target_size:
        return bias[:target_size].clone()
    resized = bias.new_zeros(target_size)
    resized[:current_size] = bias
    return resized


def _align_vocab_size(weights: OrderedDict, target_vocab_size: int) -> None:
    embedding_rows = weights["embeddings.word_embeddings.weight"].shape[0]
    decoder_rows = weights["lm_head.decoder.weight"].shape[0]
    decoder_bias_rows = weights["lm_head.decoder.bias"].shape[0]
    current_sizes = {embedding_rows, decoder_rows, decoder_bias_rows}
    if len(current_sizes) != 1:
        raise ValueError(
            "Checkpoint vocab tensors are inconsistent: "
            f"embeddings={embedding_rows}, decoder={decoder_rows}, decoder_bias={decoder_bias_rows}."
        )
    if embedding_rows != target_vocab_size:
        print(
            "Vocab size mismatch: checkpoint tensors have "
            f"{embedding_rows} rows while config expects {target_vocab_size}. Resizing embedding and MLM decoder tensors."
        )
    weights["embeddings.word_embeddings.weight"] = _resize_vocab_tensor_rows(
        weights["embeddings.word_embeddings.weight"], target_vocab_size
    )
    weights["lm_head.decoder.weight"] = _resize_vocab_tensor_rows(weights["lm_head.decoder.weight"], target_vocab_size)
    weights["lm_head.decoder.bias"] = _resize_vocab_bias(weights["lm_head.decoder.bias"], target_vocab_size)


def convert(
    path,
    version: str = "single",
    num_hidden_layers: int = 12,
    hidden_size: int = 768,
    vocab_size: int = 10,
):
    version = _validate_version(version)
    output_dir = _resolve_output_dir(path)
    if version == "single":
        config = build_config(path)
        _prepare_output_dir(output_dir, "single")
    elif version == "bpe":
        num_hidden_layers, hidden_size, vocab_size = _validate_gene_dims(num_hidden_layers, hidden_size, vocab_size)
        config = build_config_GENE(path, num_hidden_layers, hidden_size, vocab_size)
        _prepare_output_dir(output_dir, "bpe")
    elif version == "plant_bpe":
        num_hidden_layers, hidden_size, vocab_size = _validate_gene_dims(num_hidden_layers, hidden_size, vocab_size)
        config = build_config_GENE(path, num_hidden_layers, hidden_size, vocab_size)
        _prepare_output_dir(output_dir, "plant_bpe")

    config._name_or_path = str(output_dir)
    config.save_pretrained(config._name_or_path)
    ckpt = torch.load(path)
    weights = convert_ckpt(ckpt["model"])
    _align_vocab_size(weights, config.vocab_size)
    torch.save(weights, os.path.join(config._name_or_path, "pytorch_model.bin"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert UniRNA checkpoints into Hugging Face model packages.")
    parser.add_argument("checkpoint_path", help="Path to the source checkpoint file.")
    parser.add_argument(
        "--version",
        default="single",
        choices=sorted(SUPPORTED_TOKENIZER_VERSIONS),
        help="Tokenizer/config conversion preset.",
    )
    parser.add_argument("--num-hidden-layers", type=int, default=12, help="Number of transformer layers for GENE models.")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size for GENE models.")
    parser.add_argument("--vocab-size", type=int, default=10, help="Vocabulary size for GENE models.")
    return parser


def cli_main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    convert(
        args.checkpoint_path,
        version=args.version,
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    cli_main(sys.argv[1:])
