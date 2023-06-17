from transformers import EsmTokenizer


class UniRNATokenizer(EsmTokenizer):
    def _convert_token_to_id(self, token: str) -> int:
        token = token.upper() if token not in self.all_special_tokens else token
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def token_to_id(self, token: str) -> int:
        return self._convert_token_to_id(token)

    def id_to_token(self, index: int) -> str:
        return self._convert_id_to_token(index)
