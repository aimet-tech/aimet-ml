import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from aimet_ml.processing.video import convert_video, is_video, load_video

PWD = Path(__file__).parent


def validate_video(file_path: str, target_fps: Optional[int] = None) -> None:
    """
    Validate video properties based on specified criteria.

    Args:
        file_path (str): Path to the video file.
        target_fps (int, optional): The target fps used for conversion
    """
    assert is_video(file_path)

    frames, fps = load_video(file_path)

    assert isinstance(frames, list)

    if len(frames) > 0:
        assert isinstance(frames[0], np.ndarray)

    if target_fps is not None:
        assert fps == target_fps


@pytest.fixture
def audio_file_path() -> str:
    """Fixture providing the path to the audio file."""
    return str(PWD.parent.parent / "aimet_ml" / "resources" / "audios" / "sample.wav")


@pytest.fixture
def video_file_path() -> str:
    """Fixture providing the path to the video file."""
    return str(PWD.parent.parent / "aimet_ml" / "resources" / "videos" / "sample.mp4")


@pytest.fixture
def temp_output_path() -> str:
    """Fixture providing a temporary path for the output video file."""
    return str(PWD.parent.parent / "aimet_ml" / "resources" / "videos" / "output.mp4")


def test_is_video_positive(video_file_path: str) -> None:
    """Test is_video function with positive scenario."""
    assert is_video(video_file_path)


def test_is_video_negative(audio_file_path: str) -> None:
    """Test is_video function with negative scenario."""
    assert not is_video(audio_file_path)


def test_load_video(video_file_path: str) -> None:
    """Test loading video file."""
    validate_video(video_file_path)


def test_convert_video(video_file_path: str, temp_output_path: str) -> None:
    """Test converting video fps."""
    assert is_video(video_file_path)
    convert_video(video_file_path, temp_output_path, 10)
    validate_video(temp_output_path)
    os.remove(temp_output_path)
