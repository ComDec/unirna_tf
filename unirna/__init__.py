from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import UniRNAConfig, build_config
from .model import UniRNAForMaskedLM, UniRNAModels, UniRNAForSSPredict
from .tokenizer import UniRNATokenizer

__all__ = [
    "UniRNAConfig",
    "UniRNAModels",
    "UniRNAForMaskedLM",
    "UniRNAForSSPredict",
    "UniRNATokenizer",
    "build_config",
]


AutoConfig.register("unirna", UniRNAConfig)
AutoModel.register(UniRNAConfig, UniRNAModels)
AutoTokenizer.register("unirna", UniRNATokenizer)
