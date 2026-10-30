"""Local BERT tokenizer adapter for GroundingDINO."""

from pathlib import Path

import numpy as np


class BertTokenizer:
    """Minimal local tokenizer interface used by GroundingDINO inference."""

    def __init__(self, tokenizer_path):
        from tokenizers import Tokenizer

        tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        # GroundingDINO constructs its sentence masks before truncating text.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.pad_token_id = self._tokenizer.token_to_id("[PAD]")
        self._unk_token_id = self._tokenizer.token_to_id("[UNK]")

    def __call__(self, texts, *, padding="longest", return_tensors="np"):
        if padding != "longest" or return_tensors != "np":
            raise ValueError("Only longest padding and NumPy tensors are supported")
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=True)
        length = max((len(encoding.ids) for encoding in encodings), default=0)
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        result = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        for encoding in encodings:
            pad_length = length - len(encoding.ids)
            result["input_ids"].append(encoding.ids + [pad_id] * pad_length)
            result["token_type_ids"].append(encoding.type_ids + [0] * pad_length)
            result["attention_mask"].append(encoding.attention_mask + [0] * pad_length)
        return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}

    def convert_tokens_to_ids(self, tokens):
        ids = [self._tokenizer.token_to_id(token) for token in tokens]
        return [self._unk_token_id if token_id is None else token_id for token_id in ids]

    def decode(self, ids):
        return self._tokenizer.decode(ids, skip_special_tokens=False)
