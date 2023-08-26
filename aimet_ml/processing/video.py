from typing import List

import cv2
import ffmpeg
import numpy as np


def is_video(file_path: str) -> bool:
    """
    Check if a given file contains video streams.

    Args:
        file_path (str): The path to the input file.

    Returns:
        bool: True if the file contains video streams, False otherwise.
    """
    probe = ffmpeg.probe(file_path)
    streams = probe["streams"]

    for stream in streams:
        if stream["codec_type"] == "video":
            return True

    return False


def load_video(file_path: str) -> tuple:
    """
    Load frames from a video file.

    Args:
        file_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing a list of frames and the frames per second (fps).
    """
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def convert_video(src_file: str, dst_file: str, target_fps: int) -> None:
    """
    Convert a video to a different frame rate and save to a new file.

    Args:
        src_file (str): Path to the source video file.
        dst_file (str): Path to the output video file.
        target_fps (int): The target frames per second for the output video.
    """
    input_vid = ffmpeg.input(src_file)

    audio = input_vid.audio
    video = input_vid.video.filter("fps", target_fps)
    (
        ffmpeg.output(
            video,
            audio,
            dst_file,
            acodec="aac",
            loglevel="quiet",
            max_muxing_queue_size=1024,
        )
        .overwrite_output()
        .run()
    )
