import os
import shutil
import sys
from collections import OrderedDict

import torch
from chanfig import NestedDict

from unirna_tf.config import build_config, build_config_GENE


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


def convert(path, version: int = 0, num_hidden_layers: int = 12, hidden_size: int = 768, vocab_size: int = 10):
    ckpt = torch.load(path)
    if version == "single":
        config = build_config(path)
        if os.path.exists(config._name_or_path):
            shutil.rmtree(config._name_or_path)
        shutil.copytree(os.path.join(os.path.dirname(__file__), "tokenizer/single"), config._name_or_path)
    elif version == "bpe":
        config = build_config_GENE(path, num_hidden_layers, hidden_size, vocab_size)
        if os.path.exists(config._name_or_path):
            shutil.rmtree(config._name_or_path)
        shutil.copytree(os.path.join(os.path.dirname(__file__), "tokenizer/bpe"), config._name_or_path)
    elif version == "plant_bpe":
        config = build_config_GENE(path, num_hidden_layers, hidden_size, vocab_size)
        if os.path.exists(config._name_or_path):
            shutil.rmtree(config._name_or_path)
        shutil.copytree(os.path.join(os.path.dirname(__file__), "tokenizer/plant_bpe"), config._name_or_path)
    else:
        AssertionError("Invalid version for tokenizer")

    config.save_pretrained(config._name_or_path)
    ckpt = torch.load(path)
    weights = convert_ckpt(ckpt["model"])
    if config.vocab_size != weights["embeddings.word_embeddings.weight"].shape[0]:
        print(
            f"Vocab size mismatch, loaded model weights have {weights['embeddings.word_embeddings.weight'].shape[0]} tokens, while config has {config.vocab_size} tokens. \nTruncating the weights."
        )
        weights["embeddings.word_embeddings.weight"] = weights["embeddings.word_embeddings.weight"][: config.vocab_size]
    torch.save(weights, os.path.join(config._name_or_path, "pytorch_model.bin"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        convert(sys.argv[1])
    else:
        print("You choose GENE model. Please provide num_hidden_layers and hidden_size.")
        num_hidden_layers = int(input("nums_layers: "))
        hidden_size = int(input("hidden_size: "))
        vocab_size = int(input("vocab_size: "))
        convert(sys.argv[1], sys.argv[2], num_hidden_layers, hidden_size, vocab_size)
