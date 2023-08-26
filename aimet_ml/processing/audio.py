from typing import Optional, Tuple

import numpy as np
from pydub import AudioSegment, effects


def read_audio(file_path: str, target_sr: Optional[int] = None, normalize: bool = False) -> AudioSegment:
    """
    Read an audio file and return the waveform as an AudioSegment object with the target sample rate.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int, optional): Target sample rate for the audio waveform.
        normalize (bool, optional): If True, normalize the audio waveform.

    Returns:
        AudioSegment: Audio waveform as an AudioSegment object.
    """
    audio = AudioSegment.from_file(file_path)
    if target_sr:
        audio = audio.set_frame_rate(target_sr)
    if normalize:
        audio = effects.normalize(audio)
    return audio


def load_audio(file_path: str, target_sr: Optional[int] = None, normalize: bool = False) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return the waveform as a NumPy array and the target sample rate.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int, optional): Target sample rate for the audio waveform.
        normalize (bool, optional): If True, normalize the audio waveform.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the waveform as a NumPy array and the target sample rate.
    """
    audio = read_audio(file_path, target_sr, normalize)
    waveform = np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    sample_rate = audio.frame_rate
    return waveform, sample_rate


def convert_audio(
    src_file: str, dst_file: str, target_sr: Optional[int] = None, normalize: bool = False
) -> AudioSegment:
    """
    Convert an audio file to a different sample rate and save it to a new file.

    Args:
        src_file (str): Path to the source audio file.
        dst_file (str): Path to the destination audio file.
        target_sr (int, optional): Target sample rate for the output audio file.
        normalize (bool, optional): If True, normalize the audio waveform before conversion.

    Returns:
        AudioSegment: The converted audio waveform as an AudioSegment object.
    """
    audio = read_audio(src_file, target_sr, normalize)
    output_format = dst_file.split(".")[-1]
    audio.export(dst_file, format=output_format)
    return audio
