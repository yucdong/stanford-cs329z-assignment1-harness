"""Sentence chunking boundaries independent of the company document fixtures."""

import pytest

from cs329z_hw1.adapters import _chunk_doc_text
from cs329z_hw1.tokens import count_tokens


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_text_has_no_chunks(text):
    assert _chunk_doc_text(text) == []


@pytest.mark.parametrize("ending", [". ", "。"])
def test_sentences_are_packed_without_splitting(ending):
    sentence = "a" * 249 + ending
    chunks = _chunk_doc_text(sentence * 3)
    assert chunks == [(sentence * 2).strip(), sentence.strip()]


@pytest.mark.parametrize("size", [599, 600, 601, 1201])
def test_unbroken_text_obeys_hard_token_limit(size):
    text = "文" * size
    chunks = _chunk_doc_text(text)
    assert "".join(chunks) == text
    assert all(0 < count_tokens(chunk) <= 150 for chunk in chunks)


def test_long_sentence_preserves_words():
    text = " ".join(["retrieval"] * 200) + "."
    chunks = _chunk_doc_text(text)
    assert " ".join(chunks) == text
    assert all(count_tokens(chunk) <= 150 for chunk in chunks)


def test_decimal_period_does_not_end_sentence():
    first = "a" * 579 + "."
    second = "Version 3.14 is available."
    assert _chunk_doc_text(first + " " + second) == [first, second]
