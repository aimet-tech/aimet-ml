from typing import List


def include_keywords(text: str, keywords: List[str]) -> bool:
    """
    Check if any of the given keywords are present in the text.

    Args:
        text (str): The text to search for keywords.
        keywords (list[str]): List of keywords to check for.

    Returns:
        bool: True if any keyword is present in the text, False otherwise.
    """
    for kw in keywords:
        if kw in text:
            return True
    return False


def exclude_keywords(text: str, keywords: List[str]) -> bool:
    """
    Check if any of the given keywords are present in the text.

    Args:
        text (str): The text to search for keywords.
        keywords (list[str]): List of keywords to check for.

    Returns:
        bool: False if any keyword is present in the text, True otherwise.
    """
    for kw in keywords:
        if kw in text:
            return False
    return True


def clean_repeated_tokens(tokens: List[str]) -> List[str]:
    """
    Remove sequences of repeated tokens from a list.

    Args:
        tokens (list[str]): List of tokens to clean.

    Returns:
        list[str]: List of tokens with repeated sequences removed.
    """
    tokens = tokens.copy()
    sequence_size = len(tokens) // 2
    while sequence_size > 0:
        cur_idx = 0
        while cur_idx < len(tokens) - sequence_size:
            next_idx = cur_idx + sequence_size
            cur_text = "".join(tokens[cur_idx : cur_idx + sequence_size])
            next_text = "".join(tokens[next_idx : next_idx + sequence_size])
            if cur_text == next_text:
                tokens = tokens[: cur_idx + sequence_size] + tokens[next_idx + sequence_size :]
            else:
                cur_idx += 1
        sequence_size -= 1
    return tokens
