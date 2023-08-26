import pytest

from aimet_ml.utils.hamd_7 import score_to_severity


def test_normal_severity():
    """Test for scores within the 'normal' severity category."""
    assert score_to_severity(0) == "normal"
    assert score_to_severity(4) == "normal"


def test_mild_severity():
    """Test for scores within the 'mild' severity category."""
    assert score_to_severity(5) == "mild"
    assert score_to_severity(12) == "mild"


def test_moderate_severity():
    """Test for scores within the 'moderate' severity category."""
    assert score_to_severity(13) == "moderate"
    assert score_to_severity(20) == "moderate"


def test_severe_severity():
    """Test for scores within the 'severe' severity category."""
    assert score_to_severity(21) == "severe"
    assert score_to_severity(100) == "severe"


if __name__ == "__main__":
    pytest.main()
