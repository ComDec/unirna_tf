import pytest
import torch

import unirna_tf.model as model_module
from unirna_tf.model import AvgPooler, MLP, UniRNAFlashSelfAttention, UniRNAForMaskedLM, UniRNAForSSPredict, UniRNAModel


def test_unirna_model_forward_shapes(tiny_config):
    torch.manual_seed(0)
    model = UniRNAModel(tiny_config)
    input_ids = torch.tensor(
        [
            [3, 5, 7, 8, 1, 0],
            [3, 9, 8, 7, 1, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    assert outputs.last_hidden_state.shape == (2, 6, 32)
    assert outputs.pooler_output.shape == (2, 32)

    outputs_tuple = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    assert isinstance(outputs_tuple, tuple)
    assert outputs_tuple[0].shape == (2, 6, 32)


def test_masked_lm_loss_and_logits(tiny_config):
    torch.manual_seed(0)
    model = UniRNAForMaskedLM(tiny_config)
    input_ids = torch.tensor([[3, 5, 7, 8, 1, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 1] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert outputs.logits.shape == (1, 6, 10)
    assert outputs.loss is not None


def test_sspredict_outputs_and_mask(tiny_config):
    torch.manual_seed(0)
    with pytest.raises(RuntimeError):
        UniRNAForSSPredict(tiny_config)


def test_avg_pooler_excludes_special_tokens():
    pooler = AvgPooler()
    hidden_states = torch.tensor(
        [
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    pooled = pooler(hidden_states, attention_mask)
    expected = hidden_states[:, 1:-1].mean(dim=1)
    assert torch.allclose(pooled, expected)


def test_prune_heads_raises_clear_not_implemented(tiny_config):
    model = UniRNAModel(tiny_config)

    with pytest.raises(NotImplementedError, match="does not support pruning attention heads"):
        model._prune_heads({0: [0]})


def test_mlp_accepts_feature_sequences():
    mlp = MLP([1, 4, 2], residual=False)
    x = torch.randn(2, 3, 1)

    y = mlp(x)

    assert y.shape == (2, 3, 2)


def test_flash_attention_request_falls_back_when_backend_missing(monkeypatch, tiny_config):
    monkeypatch.setattr(model_module, "unirna_flash_attention", None)
    config = tiny_config
    config.use_flash_attention = True

    model = UniRNAModel(config)

    assert model.apply_flash_attention is False
    assert model.encoder.layer[0].attention.self.__class__.__name__ == "UniRNASelfAttention"


def test_flash_attention_mask_preparation_uses_bool_masks(monkeypatch, tiny_config):
    monkeypatch.setattr(model_module, "unirna_flash_attention", object())
    config = tiny_config
    config.use_flash_attention = True

    model = UniRNAModel(config)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    prepared_mask = model._prepare_attention_mask(attention_mask, input_shape=(1, 3))

    assert model.apply_flash_attention is True
    assert prepared_mask.dtype is torch.bool
    assert torch.equal(prepared_mask, attention_mask.bool())


def test_flash_attention_rejects_output_attentions_before_kernel_call(monkeypatch, tiny_config):
    monkeypatch.setattr(model_module, "unirna_flash_attention", object())
    attention = UniRNAFlashSelfAttention(tiny_config)
    hidden_states = torch.randn(1, 4, tiny_config.hidden_size)

    with pytest.raises(ValueError, match="does not support output_attentions=True"):
        attention(hidden_states, output_attentions=True)
