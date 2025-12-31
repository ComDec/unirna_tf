import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

import unirna_tf

if os.getenv("UNIRNA_SKIP_WEIGHTS") == "1":
    pytest.skip("Skipping weight integration tests in CI.", allow_module_level=True)


def test_weights_load_and_forward():
    weights_dir = Path(__file__).resolve().parents[1] / "weights" / "unirna_L8"
    if not weights_dir.exists():
        pytest.skip("weights not available")

    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    model = AutoModel.from_pretrained(weights_dir)
    inputs = tokenizer("ACGU", return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)
    assert outputs.last_hidden_state.shape[0] == 1
    assert outputs.pooler_output.shape[-1] == model.config.hidden_size

    mlm = AutoModelForMaskedLM.from_pretrained(weights_dir)
    with torch.inference_mode():
        mlm_out = mlm(**inputs)
    assert mlm_out.logits.shape[-1] == mlm.config.vocab_size

    with pytest.raises(RuntimeError):
        unirna_tf.UniRNAForSSPredict.from_pretrained(weights_dir)
