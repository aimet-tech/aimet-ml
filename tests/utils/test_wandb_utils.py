import os
from datetime import datetime

import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv

import wandb
from aimet_ml.utils.wandb_utils import list_artifact_names, load_artifact, table_to_dataframe

load_dotenv(find_dotenv())

WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_RUN_GROUP = os.getenv("WANDB_RUN_GROUP")
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TEST_TABLE_NAME = 'test-table'


@pytest.fixture
def target_df():
    return pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})


@pytest.fixture(scope='module')
def api():
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=WANDB_RUN_GROUP, name=CURRENT_TIME)
    return wandb.Api(overrides={"project": WANDB_PROJECT, "entity": WANDB_ENTITY})


@pytest.fixture(scope='module')
def target_artifact():
    return {
        'type': 'test-data',
        'collections': {
            'test-artifact': {
                'versions': {
                    'v0': {
                        'aliases': ['first'],
                        'objects': {
                            TEST_TABLE_NAME: wandb.Table(columns=['version'], data=[(0,)]),
                        },
                    },
                    'v1': {
                        'aliases': [],
                        'objects': {
                            TEST_TABLE_NAME: wandb.Table(columns=['version'], data=[(1,)]),
                        },
                    },
                    'v2': {
                        'aliases': ['latest'],
                        'objects': {
                            TEST_TABLE_NAME: wandb.Table(columns=['version'], data=[(2,)]),
                        },
                    },
                },
            },
        },
    }


@pytest.fixture
def target_artifact_names(target_artifact):
    return sorted(target_artifact['collections'].keys())


@pytest.fixture
def target_artifact_names_with_versions(target_artifact):
    return sorted(
        [
            f'{name}:{version}'
            for name in target_artifact['collections'].keys()
            for version in target_artifact['collections'][name]['versions'].keys()
        ]
    )


@pytest.fixture
def target_artifact_names_with_aliases(target_artifact):
    return sorted(
        [
            f'{name}:{alias}'
            for name in target_artifact['collections'].keys()
            for version in target_artifact['collections'][name]['versions'].keys()
            for alias in target_artifact['collections'][name]['versions'][version]['aliases']
        ]
    )


@pytest.fixture
def target_artifact_names_with_versions_and_aliases(
    target_artifact_names_with_versions, target_artifact_names_with_aliases
):
    return sorted(target_artifact_names_with_versions + target_artifact_names_with_aliases)


def extract_name_and_alias(name_with_alias, sep=':'):
    splits = name_with_alias.split(sep)
    name = sep.join(splits[:-1])
    alias = splits[-1]
    return name, alias


def get_object_by_version(target_artifact, name, version, object_path):
    return target_artifact['collections'][name]['versions'][version]['objects'][object_path]


def get_object_by_alias(target_artifact, name, alias, object_path):
    alias_to_version = {
        alias: version
        for version in sorted(target_artifact['collections'][name]['versions'].keys())
        for alias in target_artifact['collections'][name]['versions'][version]['aliases']
    }
    version = alias_to_version[alias]
    return get_object_by_version(target_artifact, name, version, object_path)


def is_df_equal(df1, df2):
    assert df1.shape == df2.shape
    assert (df1.columns == df1.columns).all()
    assert (df2.values == df2.values).all()


def test_table_to_dataframe(target_df):
    sample_table = wandb.Table(dataframe=target_df)
    sample_df = table_to_dataframe(sample_table)
    assert isinstance(sample_df, pd.DataFrame)
    is_df_equal(sample_df, target_df)


def test_list_artifact_names(api, target_artifact, target_artifact_names):
    artifact_names = list_artifact_names(api, target_artifact['type'], with_versions=False, with_aliases=False)
    assert artifact_names == target_artifact_names


def test_list_artifact_names_with_versions(api, target_artifact, target_artifact_names_with_versions):
    artifact_names = list_artifact_names(api, target_artifact['type'], with_versions=True, with_aliases=False)
    assert artifact_names == target_artifact_names_with_versions


def test_list_artifact_names_with_aliases(api, target_artifact, target_artifact_names_with_aliases):
    artifact_names = list_artifact_names(api, target_artifact['type'], with_versions=False, with_aliases=True)
    assert artifact_names == target_artifact_names_with_aliases


def test_list_artifact_names_with_versions_and_aliases(
    api, target_artifact, target_artifact_names_with_versions_and_aliases
):
    artifact_names = list_artifact_names(api, target_artifact['type'], with_versions=True, with_aliases=True)
    assert artifact_names == target_artifact_names_with_versions_and_aliases


def test_load_artifact_with_versions(api, target_artifact, target_artifact_names_with_versions):
    for name_with_version in target_artifact_names_with_versions:
        name, version = extract_name_and_alias(name_with_version)
        target_table = get_object_by_version(target_artifact, name, version, TEST_TABLE_NAME)
        loaded_table = load_artifact(api, target_artifact['type'], name, version).get(TEST_TABLE_NAME)
        assert target_table == loaded_table


def test_load_artifact_with_aliases(api, target_artifact, target_artifact_names_with_aliases):
    for name_with_alias in target_artifact_names_with_aliases:
        name, alias = extract_name_and_alias(name_with_alias)
        target_table = get_object_by_alias(target_artifact, name, alias, TEST_TABLE_NAME)
        loaded_table = load_artifact(api, target_artifact['type'], name, alias).get(TEST_TABLE_NAME)
        assert target_table == loaded_table


def test_load_artifact_with_wrong_type(api, target_artifact, target_artifact_names_with_aliases):
    name, alias = extract_name_and_alias(target_artifact_names_with_aliases[0])
    artifact = load_artifact(api, target_artifact['type'] + 'wrong for sure !!! XD 555', name, alias)
    assert artifact is None


def test_load_artifact_with_wrong_name(api, target_artifact, target_artifact_names_with_aliases):
    name, alias = extract_name_and_alias(target_artifact_names_with_aliases[0])
    artifact = load_artifact(api, target_artifact['type'], name + 'wrong for sure !!! XD 555', alias)
    assert artifact is None


def test_load_artifact_with_wrong_alias(api, target_artifact, target_artifact_names_with_aliases):
    name, alias = extract_name_and_alias(target_artifact_names_with_aliases[0])
    artifact = load_artifact(api, target_artifact['type'], name, alias + 'wrong for sure !!! XD 555')
    assert artifact is None
