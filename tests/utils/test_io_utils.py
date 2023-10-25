import json
import os
import pickle

import pytest
import yaml

from aimet_ml.utils.io_utils import read_json, read_pickle, read_yaml, write_json, write_pickle, write_yaml


@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("temp_dir")
    yield tmp_dir


@pytest.fixture
def data():
    return {
        "key1": "value1",
        "key2": {
            "key2-1": "value2-1",
            "key2-2": "value2-2",
        },
    }


def test_read_json(tmp_dir, data):
    tmp_file = os.path.join(str(tmp_dir), 'tmp.json')
    with open(tmp_file, "w") as f:
        json.dump(data, f)

    result = read_json(tmp_file)
    assert result == data


def test_read_pickle(tmp_dir, data):
    tmp_file = os.path.join(str(tmp_dir), 'tmp.pkl')
    with open(tmp_file, "wb") as f:
        pickle.dump(data, f)

    result = read_pickle(tmp_file)
    assert result == data


def test_read_yaml(tmp_dir, data):
    tmp_file = os.path.join(str(tmp_dir), 'tmp.yaml')
    with open(tmp_file, "w") as f:
        yaml.dump(data, f)

    result = read_yaml(tmp_file)
    assert result == data


def test_write_json(tmp_dir, data):
    tmp_file = os.path.join(str(tmp_dir), 'tmp.json')
    write_json(tmp_file, data)

    with open(tmp_file, "r") as f:
        result = json.load(f)

    assert result == data


def test_write_pickle(tmp_dir, data):
    tmp_file = os.path.join(str(tmp_dir), 'tmp.pkl')
    write_pickle(tmp_file, data)

    with open(tmp_file, "rb") as f:
        result = pickle.load(f)

    assert result == data


def test_write_yaml(tmp_dir, data):
    tmp_file = os.path.join(str(tmp_dir), 'tmp.yaml')
    write_yaml(tmp_file, data)

    with open(tmp_file, "r") as f:
        result = yaml.safe_load(f)

    assert result == data
