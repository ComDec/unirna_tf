import pytest

from unirna_tf.config import UniRNAConfig


@pytest.fixture()
def tiny_config():
    return UniRNAConfig(
        vocab_size=10,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        emb_layer_norm_before=True,
        token_dropout=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        use_flash_attention=False,
        tie_word_embeddings=False,
        pad_token_id=0,
        sep_token_id=1,
        cls_token_id=3,
        mask_token_id=4,
    )
