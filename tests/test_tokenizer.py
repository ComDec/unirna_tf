from pathlib import Path

import pytest
from transformers import AutoTokenizer

from unirna_tf.tokenizer import UniRNATokenizer


def _vocab_path():
    return Path(__file__).resolve().parents[1] / "weights" / "unirna_L16" / "vocab.txt"


def test_character_tokenization():
    vocab_path = _vocab_path()
    if not vocab_path.exists():
        pytest.skip("weights vocab not available")
    tokenizer = UniRNATokenizer(str(vocab_path))
    ids = tokenizer("ACGU", add_special_tokens=False)["input_ids"]
    assert ids == [5, 7, 8, 9]
    ids_lower = tokenizer("acgu", add_special_tokens=False)["input_ids"]
    assert ids_lower == ids


def test_space_separated_tokens():
    vocab_path = _vocab_path()
    if not vocab_path.exists():
        pytest.skip("weights vocab not available")
    tokenizer = UniRNATokenizer(str(vocab_path))
    ids = tokenizer("A C G U", add_special_tokens=False)["input_ids"]
    assert ids == [5, 7, 8, 9]


def test_unknown_maps_to_unk_token():
    vocab_path = _vocab_path()
    if not vocab_path.exists():
        pytest.skip("weights vocab not available")
    tokenizer = UniRNATokenizer(str(vocab_path))
    ids = tokenizer("X", add_special_tokens=False)["input_ids"]
    assert ids == [tokenizer.unk_token_id]


def test_replace_uracil_maps_to_t():
    vocab_path = _vocab_path()
    if not vocab_path.exists():
        pytest.skip("weights vocab not available")
    tokenizer = UniRNATokenizer(str(vocab_path), replace_uracil=True)
    id_u = tokenizer("U", add_special_tokens=False)["input_ids"][0]
    id_t = tokenizer("T", add_special_tokens=False)["input_ids"][0]
    assert id_u == id_t


def test_autotokenizer_unk_token():
    weights_dir = Path(__file__).resolve().parents[1] / "weights" / "unirna_L16"
    if not weights_dir.exists():
        pytest.skip("weights not available")
    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    assert tokenizer.unk_token == "N"
