from typing import Optional

import numpy as np
from pydub import AudioSegment, effects


def read_audio(file_path: str, target_sr: int = 16000, normalize: bool = False) -> tuple[np.ndarray, int]:
    """
    Read an audio file and return the waveform as an array and the target sample rate.

    Args:
        file_path (str): Path to the audio file.
        target_sr (int, optional): Target sample rate for the audio waveform.
        normalize (bool, optional): If True, normalize the audio waveform.

    Returns:
        tuple: A tuple containing the audio waveform (as a numpy array) and the target sample rate.
    """
    audio = AudioSegment.from_file(file_path).set_frame_rate(target_sr)
    if normalize:
        audio = effects.normalize(audio)
    waveform = np.asarray(audio.get_array_of_samples()).astype(np.float32, order="C") / 32768.0
    return waveform, target_sr


def convert_audio(src_file: str, dst_file: str, target_sr: Optional[int] = None) -> None:
    """
    Convert an audio file to a different sample rate and save to a new file.

    Args:
        src_file (str): Path to the source audio file.
        dst_file (str): Path to the destination audio file.
        target_sr (int, optional): Target sample rate for the output audio file.
    """
    audio = AudioSegment.from_file(src_file)
    if target_sr:
        audio = audio.set_frame_rate(target_sr)
    output_format = dst_file.split(".")[-1]
    audio.export(dst_file, format=output_format)
