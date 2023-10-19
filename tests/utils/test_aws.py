import os
from datetime import datetime

import pytest
from dotenv import find_dotenv, load_dotenv

from aimet_ml.utils.aws import download_s3, upload_dir_s3, upload_files_s3

load_dotenv(find_dotenv())

AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if AWS_S3_BUCKET is None:
    raise ValueError("AWS_S3_BUCKET environment variable is not set")


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("temp_dir")
    yield tmp_dir


def download_and_check(file_path, expected_content, tmp_dir):
    download_s3(AWS_S3_BUCKET, file_path, os.path.join(tmp_dir, os.path.basename(file_path)))
    with open(os.path.join(tmp_dir, os.path.basename(file_path)), "r") as file:
        assert file.read() == expected_content


def test_download_s3(tmp_dir):
    test_files = {
        "test_download_s3/test_file_1.txt": "hello, test file 1",
        "test_download_s3/test_file_2.txt": "hello, test file 2",
        "test_download_s3/test_file_3.txt": "hello, test file 3",
    }
    for file_path, content in test_files.items():
        download_and_check(file_path, content, tmp_dir)


def test_upload_files_s3(tmp_dir):
    unique_dir_path = f"test_upload_files/{CURRENT_TIME}"
    uploaded_files = [tmp_dir / f"test_file_{i}.txt" for i in range(1, 4)]

    upload_files_s3(AWS_S3_BUCKET, unique_dir_path, uploaded_files)

    for i in range(1, 4):
        download_and_check(f"{unique_dir_path}/test_file_{i}.txt", f"hello, test file {i}", tmp_dir)


def test_upload_dir_s3(tmp_dir):
    unique_dir_path = f"test_upload_dir/{CURRENT_TIME}"

    upload_dir_s3(AWS_S3_BUCKET, unique_dir_path, tmp_dir)

    tmp_dir_name = os.path.basename(str(tmp_dir))

    for i in range(1, 4):
        download_and_check(f"{unique_dir_path}/{tmp_dir_name}/test_file_{i}.txt", f"hello, test file {i}", tmp_dir)
