import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from aimet_ml.processing.audio import convert_audio, load_audio

PWD = Path(__file__).parent


def validate_audio(waveform: np.ndarray, sample_rate: int, target_sample_rate: Optional[int] = None) -> None:
    """
    Validate audio properties based on specified criteria.

    Args:
        waveform (np.ndarray): The audio waveform array.
        sample_rate (int): The sample rate of the audio.
        target_sample_rate (int, optional): The target sample rate used for conversion
    """
    assert isinstance(waveform, np.ndarray)
    assert np.min(waveform) >= -1.0 and np.max(waveform) <= 1.0
    if target_sample_rate is not None:
        assert isinstance(sample_rate, int)
        assert sample_rate == target_sample_rate


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
    """Fixture providing a temporary path for the output audio file."""
    return str(PWD.parent.parent / "aimet_ml" / "resources" / "audios" / "output.mp3")


def test_load_audio(audio_file_path: str) -> None:
    """Test loading audio file."""
    waveform, sample_rate = load_audio(audio_file_path)
    validate_audio(waveform, sample_rate)


def test_load_audio_with_normalize(audio_file_path: str) -> None:
    """Test the load_audio function with normalization."""
    waveform, sample_rate = load_audio(audio_file_path, normalize=True)
    validate_audio(waveform, sample_rate)


def test_load_audio_custom_sample_rate(audio_file_path: str) -> None:
    """Test the load_audio function with a custom sample rate."""
    target_sample_rate = 22050
    waveform, sample_rate = load_audio(audio_file_path, target_sample_rate)
    validate_audio(waveform, sample_rate, target_sample_rate)


def test_load_audio_custom_sample_rate_with_normalize(audio_file_path: str) -> None:
    """Test the load_audio function with a custom sample rate and normalization."""
    target_sample_rate = 44100
    waveform, sample_rate = load_audio(audio_file_path, target_sample_rate, normalize=True)
    validate_audio(waveform, sample_rate, target_sample_rate)


def test_load_audio_from_video(video_file_path: str) -> None:
    """Test loading audio from video file."""
    waveform, sample_rate = load_audio(video_file_path)
    validate_audio(waveform, sample_rate)


def test_load_audio_from_video_with_normalize(video_file_path: str) -> None:
    """Test the loading from video file with normalization."""
    waveform, sample_rate = load_audio(video_file_path, normalize=True)
    validate_audio(waveform, sample_rate)


def test_load_audio_from_video_custom_sample_rate(video_file_path: str) -> None:
    """Test the loading from video file with custom sample rate."""
    target_sample_rate = 22050
    waveform, sample_rate = load_audio(video_file_path, target_sample_rate)
    validate_audio(waveform, sample_rate, target_sample_rate)


def test_load_audio_from_video_custom_sample_rate_with_normalize(video_file_path: str) -> None:
    """Test the loading from video file with custom sample rate and normalization."""
    target_sample_rate = 44100
    waveform, sample_rate = load_audio(video_file_path, target_sample_rate, normalize=True)
    validate_audio(waveform, sample_rate, target_sample_rate)


def test_convert_audio(audio_file_path: str, temp_output_path: str) -> None:
    """Test convert audio format with custom sample rate and normalization."""
    target_sample_rate = 8000
    convert_audio(audio_file_path, temp_output_path, target_sample_rate, normalize=True)
    waveform, sample_rate = load_audio(temp_output_path)
    validate_audio(waveform, sample_rate, target_sample_rate)
    os.remove(temp_output_path)


def test_convert_audio_from_video(video_file_path: str, temp_output_path: str) -> None:
    """Test convert video to audio with custom sample rate and normalization."""
    target_sample_rate = 8000
    convert_audio(video_file_path, temp_output_path, target_sample_rate, normalize=True)
    waveform, sample_rate = load_audio(temp_output_path)
    validate_audio(waveform, sample_rate, target_sample_rate)
    os.remove(temp_output_path)
