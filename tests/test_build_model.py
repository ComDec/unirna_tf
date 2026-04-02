from pathlib import Path

import pytest
import torch

from unirna_tf.build_model import _align_vocab_size, convert, convert_ckpt


def _make_raw_model_state(vocab_size=10, hidden_size=8, num_layers=1):
    model = {
        "embed_tokens.weight": torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(vocab_size, hidden_size),
        "emb_layer_norm_before.weight": torch.ones(hidden_size),
        "emb_layer_norm_before.bias": torch.zeros(hidden_size),
        "emb_layer_norm_after.weight": torch.ones(hidden_size),
        "emb_layer_norm_after.bias": torch.zeros(hidden_size),
        "lm_head.dense.weight": torch.ones(hidden_size, hidden_size),
        "lm_head.dense.bias": torch.zeros(hidden_size),
        "lm_head.layer_norm.weight": torch.ones(hidden_size),
        "lm_head.layer_norm.bias": torch.zeros(hidden_size),
        "lm_head.out_proj.weight": torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(vocab_size, hidden_size),
        "lm_head.out_proj.bias": torch.arange(vocab_size, dtype=torch.float32),
        "layers": {},
    }
    for layer_idx in range(num_layers):
        model["layers"][str(layer_idx)] = {
            "self_attn.in_proj.weight": torch.arange(hidden_size * 3 * hidden_size, dtype=torch.float32).reshape(
                hidden_size * 3, hidden_size
            ),
            "self_attn.in_proj.bias": torch.arange(hidden_size * 3, dtype=torch.float32),
            "self_attn.rot_emb.inv_freq": torch.arange(hidden_size // 2, dtype=torch.float32),
            "self_attn.out_proj.weight": torch.ones(hidden_size, hidden_size),
            "self_attn.out_proj.bias": torch.zeros(hidden_size),
            "self_attn_layer_norm.weight": torch.ones(hidden_size),
            "self_attn_layer_norm.bias": torch.zeros(hidden_size),
            "fc1.weight": torch.ones(hidden_size * 2, hidden_size),
            "fc1.bias": torch.zeros(hidden_size * 2),
            "fc2.weight": torch.ones(hidden_size, hidden_size * 2),
            "fc2.bias": torch.zeros(hidden_size),
            "final_layer_norm.weight": torch.ones(hidden_size),
            "final_layer_norm.bias": torch.zeros(hidden_size),
        }
    return model


def test_align_vocab_size_updates_embedding_and_decoder_tensors():
    weights = convert_ckpt(_make_raw_model_state(vocab_size=10, hidden_size=8))

    _align_vocab_size(weights, 6)

    assert weights["embeddings.word_embeddings.weight"].shape == (6, 8)
    assert weights["lm_head.decoder.weight"].shape == (6, 8)
    assert weights["lm_head.decoder.bias"].shape == (6,)
    assert torch.equal(weights["embeddings.word_embeddings.weight"], torch.arange(48, dtype=torch.float32).reshape(6, 8))


def test_align_vocab_size_rejects_inconsistent_checkpoint_tensors():
    weights = convert_ckpt(_make_raw_model_state(vocab_size=10, hidden_size=8))
    weights["lm_head.decoder.bias"] = weights["lm_head.decoder.bias"][:8]

    with pytest.raises(ValueError, match="Checkpoint vocab tensors are inconsistent"):
        _align_vocab_size(weights, 8)


def test_convert_writes_output_next_to_checkpoint_and_preserves_existing_files(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "unirna_L8_H512_demo.pt"
    torch.save({"model": _make_raw_model_state(vocab_size=10, hidden_size=512)}, checkpoint_path)

    output_dir = checkpoint_dir / "unirna_L8_H512_demo"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep me")

    cwd_dir = tmp_path / "unirna_L8_H512_demo"
    cwd_dir.mkdir()
    unrelated = cwd_dir / "cwd.txt"
    unrelated.write_text("unrelated")

    monkeypatch.chdir(tmp_path)
    convert(str(checkpoint_path), version="single")

    assert (output_dir / "config.json").exists()
    assert (output_dir / "pytorch_model.bin").exists()
    assert sentinel.read_text() == "keep me"
    assert unrelated.read_text() == "unrelated"
    assert not (cwd_dir / "config.json").exists()


def test_convert_rejects_invalid_version(tmp_path):
    checkpoint_path = tmp_path / "unirna_L8_H512_demo.pt"
    torch.save({"model": _make_raw_model_state(vocab_size=10, hidden_size=512)}, checkpoint_path)

    with pytest.raises(ValueError, match="Unsupported tokenizer version"):
        convert(str(checkpoint_path), version="unknown")


def test_convert_rejects_non_positive_gene_dimensions(tmp_path):
    checkpoint_path = tmp_path / "gene.pt"
    torch.save({"model": _make_raw_model_state(vocab_size=10, hidden_size=64)}, checkpoint_path)

    with pytest.raises(ValueError, match="requires positive values"):
        convert(str(checkpoint_path), version="bpe", num_hidden_layers=0, hidden_size=64, vocab_size=10)


def test_pyproject_declares_runtime_dependencies_and_build_script():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = pyproject.read_text()

    for required in [
        '"biopython"',
        '"chanfig"',
        '"datasets"',
        '"packaging"',
        '"ray"',
        '"torch"',
        '"tqdm"',
        '"transformers"',
        'unirna_build_model = "unirna_tf.build_model:cli_main"',
        'unirna_infer = "unirna_tf.infer:cli_main"',
    ]:
        assert required in content
