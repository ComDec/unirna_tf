from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import UniRNAConfig, build_config
from .convert import convert, convert_ckpt
from .model import UniRNAForMaskedLM, UniRNAModel
from .tokenizer import UniRNATokenizer

__all__ = [
    "UniRNAConfig",
    "UniRNAModel",
    "UniRNAForMaskedLM",
    "UniRNATokenizer",
    "convert",
    "convert_ckpt",
    "build_config",
]


AutoConfig.register("unirna", UniRNAConfig)
AutoModel.register(UniRNAConfig, UniRNAModel)
AutoTokenizer.register("unirna", UniRNATokenizer)
