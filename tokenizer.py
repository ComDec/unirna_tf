from transformers import AutoTokenizer, EsmTokenizer


class UniRNATokenizer(EsmTokenizer):
    pass


AutoTokenizer.register("unirna", UniRNATokenizer)
