from typing import List

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from aimet_ml.processing.text import clean_repeated_tokens, exclude_keywords, include_keywords, trim_tokens


def tokenize(tokenizer: PreTrainedTokenizer, text: str) -> List[str]:
    """
    Tokenize the provided text using the tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to tokenize the text.
        text (str): The input text to be trimmed.

    Returns:
        List[str]: A list of tokens.
    """
    return tokenizer.tokenize(text)


def trim_and_compare(tokenizer: PreTrainedTokenizer, text: str, max_len: int, expected_len: int):
    """
    Trims a text to the specified maximum length using a tokenizer and compares the results with expectations.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to tokenize the text.
        text (str): The input text to be trimmed.
        max_len (int): The maximum length for the trimmed text.
        expected_len (int): The expected length of the trimmed text.
    """
    trimmed_text, num_tokens = trim_tokens(tokenizer, text, max_len)
    trimmed_tokens = tokenize(tokenizer, trimmed_text)
    assert trimmed_text in text
    assert len(trimmed_tokens) == num_tokens
    assert num_tokens == expected_len


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    """Get sample tokenizer for testing."""
    return AutoTokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2")


@pytest.fixture
def sample_text() -> str:
    """Get sample text for testing."""
    return "unit testing is a fundamental practice in software development"


def test_include_keywords_positive():
    """Test include_keywords function with positive scenario."""
    text = "This is a sample text containing the word apple."
    keywords = ["apple", "banana", "cherry"]
    assert include_keywords(text, keywords) is True


def test_include_keywords_negative():
    """Test include_keywords function with negative scenario."""
    text = "This is a sample text."
    keywords = ["apple", "banana", "cherry"]
    assert include_keywords(text, keywords) is False


def test_exclude_keywords_positive():
    """Test exclude_keywords function with positive scenario."""
    text = "This is a sample text without the word."
    keywords = ["apple", "banana", "cherry"]
    assert exclude_keywords(text, keywords) is True


def test_exclude_keywords_negative():
    """Test exclude_keywords function with negative scenario."""
    text = "This is a sample text containing the word apple."
    keywords = ["apple", "banana", "cherry"]
    assert exclude_keywords(text, keywords) is False


def test_clean_repeated_tokens():
    """Test clean_repeated_tokens function."""
    tokens = ["hello", "world", "world", "world", "python", "python", "code"]
    cleaned_tokens = clean_repeated_tokens(tokens)
    assert cleaned_tokens == ["hello", "world", "python", "code"]


def test_clean_repeated_tokens_empty():
    """Test clean_repeated_tokens function with an empty list of tokens."""
    tokens = []
    cleaned_tokens = clean_repeated_tokens(tokens)
    assert cleaned_tokens == []


def test_clean_repeated_tokens_single_token():
    """Test clean_repeated_tokens function with a list containing a single token."""
    tokens = ["hello"]
    cleaned_tokens = clean_repeated_tokens(tokens)
    assert cleaned_tokens == ["hello"]


def test_trim_tokens_with_longer_len(tokenizer: PreTrainedTokenizer, sample_text: str):
    """
    Test trim_and_compare function with max_len longer than the token length of sample_text.

    Args:
        tokenizer (PretrainedTokenizer): The tokenizer.
        sample_text (str): The sample text.
    """
    sample_token_len = len(tokenize(tokenizer, sample_text))
    trim_and_compare(tokenizer, sample_text, 2 * sample_token_len, sample_token_len)


def test_trim_tokens_with_same_len(tokenizer: PreTrainedTokenizer, sample_text: str):
    """
    Test trim_and_compare function with max_len equal to the token length of sample_text.

    Args:
        tokenizer (PretrainedTokenizer): The tokenizer.
        sample_text (str): The sample text.
    """
    sample_token_len = len(tokenize(tokenizer, sample_text))
    trim_and_compare(tokenizer, sample_text, sample_token_len, sample_token_len)


def test_trim_tokens_with_half_len(tokenizer: PreTrainedTokenizer, sample_text: str):
    """
    Test trim_and_compare function with max_len being half the token length of sample_text.

    Args:
        tokenizer (PretrainedTokenizer): The tokenizer.
        sample_text (str): The sample text.
    """
    sample_token_len = len(tokenize(tokenizer, sample_text))
    trim_and_compare(tokenizer, sample_text, sample_token_len // 2, sample_token_len // 2)


def test_trim_tokens_with_zero_len(tokenizer: PreTrainedTokenizer, sample_text: str):
    """
    Test trim_and_compare function with max_len equal to zero.

    Args:
        tokenizer (PretrainedTokenizer): The tokenizer.
        sample_text (str): The sample text.
    """
    trim_and_compare(tokenizer, sample_text, 0, 0)


if __name__ == "__main__":
    pytest.main()
