# coding=utf-8
"""Chinese-CLIP BERT tokenizer, copied from app/domain/common/tokenizer.py (Python 3 only, no six)."""

import collections
import unicodedata
from pathlib import Path
from typing import List, Union

VOCAB_FILE = Path(__file__).resolve().parent / "models/clip/vocab.txt"


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        for index, token in enumerate(reader):
            vocab[token.strip()] = index
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


class BasicTokenizer(object):
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)

        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        return whitespace_tokenize(" ".join(split_tokens))

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        return "".join(char for char in text if unicodedata.category(char) != "Mn")

    def _run_split_on_punc(self, text):
        output = []
        start_new = True
        for char in text:
            if _is_punctuation(char):
                output.append([char])
                start_new = True
            else:
                if start_new:
                    output.append([])
                start_new = False
                output[-1].append(char)
        return ["".join(x) for x in output]

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            output.append(" " if _is_whitespace(char) else char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            if _is_chinese_char(ord(char)):
                output.extend([" ", char, " "])
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class FullTokenizer(object):
    def __init__(self, vocab_file=VOCAB_FILE, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for tok in self.basic_tokenizer.tokenize(text):
            split_tokens.extend(self.wordpiece_tokenizer.tokenize(tok))
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[t] for t in tokens]


def _is_whitespace(char):
    return char in " \t\n\r" or unicodedata.category(char) == "Zs"


def _is_control(char):
    if char in "\t\n\r":
        return False
    return unicodedata.category(char) in ("Cc", "Cf")


def _is_punctuation(char):
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _is_chinese_char(cp):
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


_tokenizer = FullTokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 52) -> List[List[int]]:
    """Wrap each text in [CLS]/[SEP], truncate to context_length and pad with 0."""
    if isinstance(texts, str):
        texts = [texts]
    cls_id = _tokenizer.vocab["[CLS]"]
    sep_id = _tokenizer.vocab["[SEP]"]
    results = []
    for text in texts:
        token_ids = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[: context_length - 2]
        ids = [cls_id] + token_ids + [sep_id]
        results.append(ids + [0] * (context_length - len(ids)))
    return results
