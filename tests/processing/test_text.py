import pytest

from aimet_ml.processing.text import clean_repeated_tokens, exclude_keywords, include_keywords


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


# Run the tests
if __name__ == "__main__":
    pytest.main()
