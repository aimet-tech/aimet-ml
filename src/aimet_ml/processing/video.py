import cv2
import ffmpeg


def is_video(filename: str) -> bool:
    """
    Check if a given file contains video streams.

    Parameters:
        filename (str): The path to the input file.

    Returns:
        bool: True if the file contains video streams, False otherwise.
    """
    probe = ffmpeg.probe(filename)
    streams = probe["streams"]

    for stream in streams:
        if stream["codec_type"] == "video":
            return True

    return False


def load_video(video_path: str, verbose: bool = False) -> tuple:
    """
    Load frames from a video file.

    Parameters:
        video_path (str): The path to the video file.
        verbose (bool, optional): If True, print frame loading information.

    Returns:
        tuple: A tuple containing a list of frames and the frames per second (fps).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if verbose:
            print(len(frames), type(frame), ret)
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def convert_video(vid_path: str, output_path: str, target_fps: int = None) -> None:
    """
    Convert a video to a different frame rate and save to a new file.

    Parameters:
        vid_path (str): The path to the input video file.
        output_path (str): The path to the output video file.
        target_fps (int, optional): The target frames per second for the output video.
    """
    input_vid = ffmpeg.input(vid_path)

    if target_fps:
        audio = input_vid.audio
        video = input_vid.video.filter("fps", target_fps)
        (
            ffmpeg.output(
                video,
                audio,
                output_path,
                acodec="aac",
                loglevel="quiet",
                max_muxing_queue_size=1024,
            )
            .overwrite_output()
            .run()
        )
    else:
        (
            ffmpeg.output(
                input_vid,
                output_path,
                acodec="aac",
                loglevel="quiet",
                max_muxing_queue_size=1024,
            )
            .overwrite_output()
            .run()
        )
