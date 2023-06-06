from transformers import AutoModel, EsmModel, EsmForMaskedLM


class UniRNAModel(EsmModel):
    pass


class UniRNAForMaskedLM(EsmForMaskedLM):
    pass


AutoModel.register("unirna", UniRNAModel)
