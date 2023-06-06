from transformers import AutoModel, EsmModel, EsmForMaskedLM

from config import UniRNAConfig


class AveragePooler:
    def __call__(self, hidden_states):
        return hidden_states.mean(dim=1)


class ClsPooler:
    def __call__(self, hidden_states):
        return hidden_states[:, 0]


class UniRNAModel(EsmModel):
    config_class = UniRNAConfig
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.pooler
        del self.contact_head
        self.pooler = ClsPooler()


class UniRNAForMaskedLM(EsmForMaskedLM):
    config_class = UniRNAConfig


AutoModel.register(UniRNAConfig, UniRNAModel)
