"""Check tokenizer output against known BERT fast-tokenizer results."""

import unittest
from pathlib import Path

import numpy as np

from app.domain.groundingdino.preprocess import prepare_text_inputs
from app.domain.groundingdino.tokenizer import BertTokenizer
from scripts.infer_folder import BertTokenizer as StandaloneBertTokenizer
from scripts.infer_folder import prepare_text_inputs as standalone_prepare_text_inputs

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "config/model/groundingdino/bert-base-uncased"


class TokenizerChecks(unittest.TestCase):
    def test_known_encodings_and_decoding(self):
        cases = [
            ("glass . straw . lemon .", [101, 3221, 1012, 13137, 1012, 14380, 1012, 102],
             "[CLS] glass. straw. lemon. [SEP]"),
            ("Caf\u00e9! don't, [UNK]", [101, 7668, 999, 2123, 1005, 1056, 1010, 100, 102],
             "[CLS] cafe! don ' t, [UNK] [SEP]"),
            ("supercalifragilisticexpialidocious",
             [101, 3565, 9289, 10128, 29181, 24411, 4588, 10288, 19312, 21273, 10085, 6313, 102],
             "[CLS] supercalifragilisticexpialidocious [SEP]"),
        ]
        for tokenizer_class in (BertTokenizer, StandaloneBertTokenizer):
            tokenizer = tokenizer_class(TOKENIZER_PATH)
            for text, ids, decoded in cases:
                with self.subTest(tokenizer=tokenizer_class, text=text):
                    encoded = tokenizer([text])
                    self.assertEqual(encoded["input_ids"].tolist(), [ids])
                    self.assertEqual(encoded["token_type_ids"].tolist(), [[0] * len(ids)])
                    self.assertEqual(encoded["attention_mask"].tolist(), [[1] * len(ids)])
                    self.assertEqual(tokenizer.decode(ids), decoded)
            self.assertEqual(tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?", "missing-token"]),
                             [101, 102, 1012, 1029, 100])
            batch = tokenizer(["glass", "glass straw"])
            self.assertEqual(batch["input_ids"].tolist(), [[101, 3221, 102, 0], [101, 3221, 13137, 102]])
            self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 0], [1, 1, 1, 1]])

    def test_text_masks_truncation_and_standalone_match(self):
        caption = "glass . straw ? lemon " * 80
        shared, shared_metadata = prepare_text_inputs(caption, BertTokenizer(TOKENIZER_PATH),
                                                     max_text_len=256, fixed_text_len=32)
        standalone, standalone_metadata = standalone_prepare_text_inputs(
            caption, StandaloneBertTokenizer(TOKENIZER_PATH), max_text_len=256, fixed_text_len=32)
        self.assertGreater(shared_metadata["token_count_before_truncation"], 256)
        self.assertEqual(shared_metadata["text_length"], 32)
        self.assertEqual(shared_metadata, standalone_metadata)
        for name in shared:
            np.testing.assert_array_equal(shared[name], standalone[name])
        self.assertEqual(shared["position_ids"][0, :8].tolist(), [0, 0, 1, 0, 1, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
