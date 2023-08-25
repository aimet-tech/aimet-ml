def score_to_severity(score: int) -> str:
    """
    Convert a score to a severity category.

    Args:
        score (int): The input score.

    Returns:
        str: The corresponding severity category ("normal", "mild", "moderate", or "severe").
    """
    if score < 5:
        return "normal"
    if score < 13:
        return "mild"
    if score < 21:
        return "moderate"
    return "severe"
