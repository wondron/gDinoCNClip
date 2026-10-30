# coding=utf-8
# BERT tokenizer + 简化版 tokenize（无 torch）

from __future__ import absolute_import, division, print_function

import collections
import re
import unicodedata
import six
from functools import lru_cache
import os
from typing import Union, List
import logging

logger = logging.getLogger("tokenizer")

@lru_cache()
def default_vocab():
    vvocation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.txt")
    logger.info(vvocation_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.txt")


def convert_to_unicode(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):  # noqa: F821
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


# =======================
# Tokenizer classes
# =======================

class BasicTokenizer(object):
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = convert_to_unicode(text)
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
        output = []
        for char in text:
            if unicodedata.category(char) != "Mn":
                output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        output = []
        start_new = True
        for char in chars:
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
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp):
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
        text = convert_to_unicode(text)
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
    def __init__(self, vocab_file=default_vocab(), do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for tok in self.basic_tokenizer.tokenize(text):
            split_tokens.extend(self.wordpiece_tokenizer.tokenize(tok))
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[i] for i in ids]


# Helper functions -----------------------------

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

_tokenizer = FullTokenizer(vocab_file=default_vocab(), do_lower_case=True)

def tokenize(texts: Union[str, List[str]], context_length: int = 52) -> List[List[int]]:
    """
    返回 list[list[int]]，长度统一为 context_length
    用 CLS / SEP 包裹，超长截断，不足 padding 0
    """
    if isinstance(texts, str):
        texts = [texts]

    results = []

    cls_id = _tokenizer.vocab["[CLS]"]
    sep_id = _tokenizer.vocab["[SEP]"]

    for text in texts:
        tokens = _tokenizer.tokenize(text)
        token_ids = _tokenizer.convert_tokens_to_ids(tokens)

        token_ids = token_ids[: context_length - 2]

        ids = [cls_id] + token_ids + [sep_id]

        # padding
        if len(ids) < context_length:
            ids = ids + [0] * (context_length - len(ids))

        results.append(ids)

    return results
