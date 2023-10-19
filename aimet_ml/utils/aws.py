import os

import boto3
from tqdm import tqdm


def download_s3(bucket_name: str, object_name: str, output_path: str):
    """
    Download a file from an S3 bucket and save it to the local file system.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_name (str): The key or path of the object to be downloaded from the bucket.
        output_path (str): The local file path to save the downloaded object.
    """
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, object_name, output_path)


def upload_files_s3(bucket_name: str, bucket_dir_path: str, src_file_paths: list):
    """
    Upload multiple local files to an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        bucket_dir_path (str): The path within the bucket where the files will be uploaded.
        src_file_paths (list): A list of local file paths to be uploaded to the bucket.
    """
    s3 = boto3.client("s3")
    for file_path in tqdm(src_file_paths):
        object_name = os.path.join(bucket_dir_path, os.path.basename(file_path))
        s3.upload_file(file_path, bucket_name, object_name)


def upload_dir_s3(bucket_name: str, bucket_dir_path: str, src_dir_path: str):
    """
    Upload a local directory to an S3 bucket, preserving the directory structure.

    Args:
        bucket_name (str): The name of the S3 bucket.
        bucket_dir_path (str): The path within the bucket where the local directory will be uploaded.
        src_dir_path (str): The local directory path to be uploaded.
    """
    s3 = boto3.client("s3")
    src_dir_name = os.path.basename(src_dir_path)
    for r, _, f in tqdm(os.walk(src_dir_path)):
        for n in f:
            file_path = os.path.join(r, n)
            relpath = os.path.relpath(r, src_dir_path)
            object_name = os.path.join(bucket_dir_path, src_dir_name, "" if relpath == "." else relpath, n)
            s3.upload_file(file_path, bucket_name, object_name)
