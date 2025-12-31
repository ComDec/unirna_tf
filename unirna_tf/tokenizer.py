# Copyright Beijing DP Technology Co.,Ltd. All rights reserved.

"""
    The code is modified from the original ESM tokenizer provided by HuggingFace.
    Sources: https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/tokenization_esm.py
"""

import os
from typing import List, Optional, Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocab_file(vocab_file):
    """Load vocabulary tokens from file into a list of strings."""
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [line.strip() for line in lines]


class UniRNATokenizer(PreTrainedTokenizer):
    """
    Constructs an UniRNA tokenizer, based on ESM tokenizer provided by HuggingFace.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="N",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        replace_uracil: bool = False,
        **kwargs,
    ):
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}
        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )

        # Optional compatibility switch for DNA-only workflows.
        if replace_uracil and "U" in self._token_to_id and "T" in self._token_to_id:
            self._token_to_id["U"] = self._token_to_id["T"]
        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    def _convert_token_to_id(self, token: str) -> int:
        token = token.upper() if token not in self.all_special_tokens else token
        unk_id = self._token_to_id.get(self.unk_token)
        if unk_id is None:
            unk_id = self.unk_token_id
        return self._token_to_id.get(token, unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def token_to_id(self, token: str) -> int:
        return self._convert_token_to_id(token)

    def id_to_token(self, index: int) -> str:
        return self._convert_id_to_token(index)

    def _tokenize(self, text, **kwargs):
        text = text.strip()
        if not text:
            return []
        if any(ch.isspace() for ch in text):
            return text.split()
        return list(text)

    def get_vocab_size(self, with_added_tokens=False):
        if with_added_tokens:
            return len(self.get_vocab())
        return len(self._id_to_token)

    def get_vocab(self):
        vocab = {token: i for i, token in enumerate(self.all_tokens)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + sep  # Multiple inputs always have an EOS token

    def get_special_tokens_mask(
        self,
        token_ids_0: List,
        token_ids_1: Optional[List] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt",
        )
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size(with_added_tokens=False)

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        return super()._add_tokens(new_tokens, special_tokens=special_tokens)
