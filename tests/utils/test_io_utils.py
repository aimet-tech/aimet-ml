import json
import pickle
from pathlib import Path

import pytest
import yaml

from aimet_ml.utils.io_utils import read_json, read_pickle, read_yaml, write_json, write_pickle, write_yaml


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Create a temporary directory using pytest's tmp_path_factory.

    Args:
        tmp_path_factory (pytest.TempPathFactory): The pytest tmp_path_factory.

    Yields:
        Path: The path to the temporary directory.
    """
    tmp_dir = tmp_path_factory.mktemp("temp_dir")
    yield tmp_dir


@pytest.fixture
def data() -> dict:
    """
    Generate a sample data dictionary for testing.

    Returns:
        dict: A sample data dictionary.
    """
    return {
        "key1": "value1",
        "key2": {
            "key2-1": "value2-1",
            "key2-2": "value2-2",
        },
    }


def test_read_json(tmp_dir: Path, data: dict):
    """
    Test reading data from a JSON file.

    Args:
        tmp_dir (Path): The temporary directory where the file will be created.
        data (dict): The data to be written to the JSON file.

    """
    tmp_file = tmp_dir / 'tmp.json'
    with open(tmp_file, "w") as f:
        json.dump(data, f)

    result = read_json(tmp_file)
    assert result == data


def test_read_pickle(tmp_dir: Path, data: dict):
    """
    Test reading data from a Pickle file.

    Args:
        tmp_dir (Path): The temporary directory where the file will be created.
        data (dict): The data to be written to the Pickle file.
    """
    tmp_file = tmp_dir / 'tmp.pkl'
    with open(tmp_file, "wb") as f:
        pickle.dump(data, f)

    result = read_pickle(tmp_file)
    assert result == data


def test_read_yaml(tmp_dir: Path, data: dict):
    """
    Test reading data from a YAML file.

    Args:
        tmp_dir (Path): The temporary directory where the file will be created.
        data (dict): The data to be written to the YAML file.
    """
    tmp_file = tmp_dir / 'tmp.yaml'
    with open(tmp_file, "w") as f:
        yaml.dump(data, f)

    result = read_yaml(tmp_file)
    assert result == data


def test_write_json(tmp_dir: Path, data: dict):
    """
    Test writing data to a JSON file and then reading it back.

    Args:
        tmp_dir (Path): The temporary directory where the file will be created.
        data (dict): The data to be written to the JSON file.
    """
    tmp_file = tmp_dir / 'tmp.json'
    write_json(tmp_file, data)

    with open(tmp_file, "r") as f:
        result = json.load(f)

    assert result == data


def test_write_pickle(tmp_dir: Path, data: dict):
    """
    Test writing data to a Pickle file and then reading it back.

    Args:
        tmp_dir (Path): The temporary directory where the file will be created.
        data (dict): The data to be written to the Pickle file.
    """
    tmp_file = tmp_dir / 'tmp.pkl'
    write_pickle(tmp_file, data)

    with open(tmp_file, "rb") as f:
        result = pickle.load(f)

    assert result == data


def test_write_yaml(tmp_dir: Path, data: dict):
    """
    Test writing data to a YAML file and then reading it back.

    Args:
        tmp_dir (Path): The temporary directory where the file will be created.
        data (dict): The data to be written to the YAML file.
    """
    tmp_file = tmp_dir / 'tmp.yaml'
    write_yaml(tmp_file, data)

    with open(tmp_file, "r") as f:
        result = yaml.safe_load(f)

    assert result == data
