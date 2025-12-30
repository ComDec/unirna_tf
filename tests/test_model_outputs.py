import torch

from unirna_tf.model import AvgPooler, UniRNAForMaskedLM, UniRNAForSSPredict, UniRNAModel


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
    model = UniRNAForSSPredict(tiny_config)
    input_ids = torch.tensor([[3, 5, 7, 8, 9, 1]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0, 1, 1]], dtype=torch.long)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    assert outputs.logits.shape == (1, 4, 4, 1)
    assert outputs.pair_mask is not None
    assert outputs.pair_mask.shape == outputs.logits.shape

    masked_logits = outputs.logits[~outputs.pair_mask]
    assert torch.allclose(masked_logits, torch.zeros_like(masked_logits))


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
