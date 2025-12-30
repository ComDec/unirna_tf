import os

from transformers import PretrainedConfig


class UniRNAConfig(PretrainedConfig):
    model_type: str = "unirna"

    def __init__(
        self,
        vocab_size: int = 10,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        max_position_embeddings: int = 1026,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        sep_token_id: int = 1,
        cls_token_id: int = 3,
        mask_token_id: int = 4,
        emb_layer_norm_before: bool = True,
        token_dropout: bool = True,
        position_embedding_type: str = "rotary",
        use_flash_attention: bool = False,
        tie_word_embeddings: bool = False,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            sep_token_id=sep_token_id,
            cls_token_id=cls_token_id,
            mask_token_id=mask_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=is_decoder,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.position_embedding_type = position_embedding_type
        self.use_flash_attention = use_flash_attention
        if self.architectures is None:
            self.architectures = ["UniRNAForMaskedLM"]


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
        cls_token_id=3,
        vocab_size=10,
        emb_layer_norm_before=True,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        token_dropout=True,
        initializer_range=0.02,
        use_flash_attention=True,
        max_position_embeddings=1026,
        position_embedding_type="rotary",
        tie_word_embeddings=False,
    )
    config._name_or_path = name
    return config


def build_config_GENE(path, num_hidden_layers: int, hidden_size: int, vocab_size: int, model_type="GENE"):
    path = os.path.splitext(path)[0]
    name = os.path.basename(path)
    # model_type, num_hidden_layers, hidden_size, _ = name.split("_")[:4]
    num_hidden_layers = int(num_hidden_layers)
    hidden_size = int(hidden_size)
    num_attention_heads = hidden_size // 64
    intermediate_size = hidden_size * 4
    config = UniRNAConfig(
        model_type=model_type,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        pad_token_id=0,
        sep_token_id=1,
        mask_token_id=4,
        cls_token_id=3,
        vocab_size=vocab_size,
        emb_layer_norm_before=True,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        token_dropout=True,
        initializer_range=0.02,
        use_flash_attention=True,
        max_position_embeddings=1026,
        position_embedding_type="rotary",
        tie_word_embeddings=False,
    )
    config._name_or_path = name
    return config
