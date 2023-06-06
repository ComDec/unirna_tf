import os

from transformers import AutoConfig, PretrainedConfig


class UniRNAConfig(PretrainedConfig):
    model_type: str = "unirna"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architectures = ["UniRNAForMaskedLM"]
        self.position_embedding_type = "rotary"


AutoConfig.register("unirna", UniRNAConfig)


def build_config(path):
    path = os.path.splitext(path)[0]
    name = os.path.basename(path)
    model_type, num_hidden_layers, hidden_size, _ = name.split("_")[:4]
    num_hidden_layers = int(num_hidden_layers[1:])
    hidden_size = int(hidden_size[1:])
    num_attention_heads = hidden_size // 64
    intermediate_size = hidden_size * 3
    config = UniRNAConfig(
        model_type=model_type,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        pad_token_id=0,
        sep_token_id=1,
        mask_token_id=4,
        vocab_size=10,
        emb_layer_norm_before=True,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=1026,
        token_dropout=True,
        initializer_range=0.02,
    )
    config._name_or_path = name
    return config
