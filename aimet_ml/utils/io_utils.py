import json
import pickle
from typing import Optional

import yaml


def read_json(file_path: str) -> dict:
    """
    Read and parse a JSON file.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        dict: A dictionary containing the parsed JSON data.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def read_pickle(file_path: str):
    """
    Read and unpickle a binary pickle file.

    Args:
        file_path (str): The path to the pickle file to be read.

    Returns:
        Any: The unpickled object.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def read_yaml(file_path: str) -> dict:
    """
    Read and parse a YAML file.

    Args:
        file_path (str): The path to the YAML file to be read.

    Returns:
        dict: A dictionary containing the parsed YAML data.
    """
    with open(file_path, "r") as f:
        result = yaml.safe_load(f)
    return result


def write_json(file_path: str, data: dict, indent: Optional[int] = None):
    """
    Write data to a JSON file.

    Args:
        file_path (str): The path to the JSON file to be written.
        data (dict): The data to be written to the JSON file.
        indent (int, optional): The number of spaces to use for indentation.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def write_pickle(file_path: str, data):
    """
    Write data to a binary pickle file.

    Args:
        file_path (str): The path to the pickle file to be written.
        data: The data to be pickled and written to the file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def write_yaml(file_path: str, data: dict, default_flow_style: bool = False):
    """
    Write data to a YAML file.

    Args:
        file_path (str): The path to the YAML file to be written.
        data (dict): The data to be written to the YAML file.
        default_flow_style (bool, optional): Whether to use the default flow style for YAML.
    """
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=default_flow_style)
