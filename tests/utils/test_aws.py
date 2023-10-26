import os
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

from aimet_ml.utils.aws import download_s3, upload_dir_s3, upload_files_s3

load_dotenv(find_dotenv(), override=True)

AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if AWS_S3_BUCKET is None:
    raise ValueError("AWS_S3_BUCKET environment variable is not set")


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory: pytest.TempPathFactory):
    """
    Fixture for creating a temporary directory using pytest's TempPathFactory.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Pytest fixture for creating temporary directories.

    Returns:
        pathlib.Path: The temporary directory path.
    """
    tmp_dir = tmp_path_factory.mktemp("temp_dir")
    yield tmp_dir


def download_and_check(file_path: str, expected_content: str, tmp_dir: Path):
    """
    Downloads a file from AWS S3 and checks its content against the expected content.

    Args:
        file_path (str): The file path on AWS S3.
        expected_content (str): The expected content of the file.
        tmp_dir (pathlib.Path): The local temporary directory where the file will be downloaded.

    Raises:
        AssertionError: If the downloaded file content does not match the expected content.
    """
    output_path = os.path.join(tmp_dir, os.path.basename(file_path))
    download_s3(str(AWS_S3_BUCKET), file_path, output_path)
    with open(output_path, "r") as file:
        assert file.read() == expected_content


def test_download_s3(tmp_dir: Path):
    """
    Test function for downloading files from AWS S3 and checking their content.

    Args:
        tmp_dir (pathlib.Path): The local temporary directory for storing downloaded files.
    """
    test_files = {
        "test_download_s3/test_file_1.txt": "hello, test file 1",
        "test_download_s3/test_file_2.txt": "hello, test file 2",
        "test_download_s3/test_file_3.txt": "hello, test file 3",
    }
    for file_path, content in test_files.items():
        download_and_check(file_path, content, tmp_dir)


def test_upload_files_s3(tmp_dir: Path):
    """
    Test function for uploading files to AWS S3 and then downloading and checking their content.

    Args:
        tmp_dir (pathlib.Path): The local temporary directory for storing uploaded and downloaded files.
    """
    unique_dir_path = f"test_upload_files/{CURRENT_TIME}"
    uploaded_files = [str(tmp_dir / f"test_file_{i}.txt") for i in range(1, 4)]

    upload_files_s3(str(AWS_S3_BUCKET), unique_dir_path, uploaded_files)

    for i in range(1, 4):
        download_and_check(f"{unique_dir_path}/test_file_{i}.txt", f"hello, test file {i}", tmp_dir)


def test_upload_dir_s3(tmp_dir: Path):
    """
    Test function for uploading a directory to AWS S3 and then downloading and checking the content.

    Args:
        tmp_dir (pathlib.Path): The local temporary directory for storing uploaded and downloaded files.
    """
    unique_dir_path = f"test_upload_dir/{CURRENT_TIME}"

    upload_dir_s3(str(AWS_S3_BUCKET), unique_dir_path, str(tmp_dir))

    tmp_dir_name = os.path.basename(tmp_dir)

    for i in range(1, 4):
        download_and_check(f"{unique_dir_path}/{tmp_dir_name}/test_file_{i}.txt", f"hello, test file {i}", tmp_dir)
