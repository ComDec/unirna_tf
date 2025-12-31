from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from .config import UniRNAConfig, build_config
from .model import UniRNAForMaskedLM, UniRNAForSSPredict, UniRNAModels
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
AutoModelForMaskedLM.register(UniRNAConfig, UniRNAForMaskedLM)
AutoTokenizer.register("unirna", UniRNATokenizer)
