import os
from datetime import datetime

import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv

import wandb
from aimet_ml.utils.wandb_utils import list_artifact_names, load_artifact, table_to_dataframe

load_dotenv(find_dotenv(), override=True)

WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_RUN_GROUP = os.getenv("WANDB_RUN_GROUP")
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TEST_TABLE_NAME = 'test-table'


@pytest.fixture
def target_df() -> pd.DataFrame:
    """
    Fixture that returns a sample target DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with sample data.
    """
    return pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})


@pytest.fixture(scope='module')
def api() -> wandb.Api:
    """
    Fixture that initializes the WandB API and returns it.

    Returns:
        wandb.Api: The initialized WandB API.
    """
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=WANDB_RUN_GROUP, name=CURRENT_TIME)
    return wandb.Api(overrides={"project": WANDB_PROJECT, "entity": WANDB_ENTITY})


@pytest.fixture(scope='module')
def target_artifact() -> dict:
    """
    Fixture that returns a sample target artifact.

    Returns:
        dict: A sample WandB artifact dictionary.
    """
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
def target_artifact_names(target_artifact: dict) -> list:
    """
    Fixture that returns the names of target artifacts.

    Args:
        target_artifact (dict): The target WandB artifact dictionary.

    Returns:
        list: A sorted list of artifact names.
    """
    return sorted(target_artifact['collections'].keys())


@pytest.fixture
def target_artifact_names_with_versions(target_artifact: dict) -> list:
    """
    Fixture that returns the names of target artifacts with versions.

    Args:
        target_artifact (dict): The target WandB artifact dictionary.

    Returns:
        list: A sorted list of artifact names with versions.
    """
    return sorted(
        [
            f'{name}:{version}'
            for name in target_artifact['collections'].keys()
            for version in target_artifact['collections'][name]['versions'].keys()
        ]
    )


@pytest.fixture
def target_artifact_names_with_aliases(target_artifact: dict) -> list:
    """
    Fixture that returns the names of target artifacts with aliases.

    Args:
        target_artifact (dict): The target WandB artifact dictionary.

    Returns:
        list: A sorted list of artifact names with aliases.
    """
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
    target_artifact_names_with_versions: list, target_artifact_names_with_aliases: list
) -> list:
    """
    Fixture that returns the names of target artifacts with versions and aliases.

    Args:
        target_artifact_names_with_versions (list): A list of artifact names with versions.
        target_artifact_names_with_aliases (list): A list of artifact names with aliases.

    Returns:
        list: A sorted list of artifact names with versions and aliases.
    """
    return sorted(target_artifact_names_with_versions + target_artifact_names_with_aliases)


def extract_name_and_alias(name_with_alias: str, sep: str = ':') -> tuple:
    """
    Extracts the name and alias from a combined name_with_alias string.

    Args:
        name_with_alias (str): The combined name and alias string.
        sep (str): The separator between the name and alias.

    Returns:
        tuple: A tuple containing the extracted name and alias.
    """
    splits = name_with_alias.split(sep)
    name = sep.join(splits[:-1])
    alias = splits[-1]
    return name, alias


def get_object_by_version(target_artifact: dict, name: str, version: str, object_path: str) -> wandb.Table:
    """
    Get an object from a target artifact by specifying its name, version, and object path.

    Args:
        target_artifact (dict): The target artifact.
        name (str): The name of the object collection.
        version (str): The version of the object.
        object_path (str): The path to the object.

    Returns:
        wandb.Table: The retrieved object.
    """
    return target_artifact['collections'][name]['versions'][version]['objects'][object_path]


def get_object_by_alias(target_artifact: dict, name: str, alias: str, object_path: str) -> wandb.Table:
    """
    Get an object from a target artifact by specifying its name, alias, and object path.

    Args:
        target_artifact (dict): The target artifact.
        name (str): The name of the object collection.
        alias (str): The alias of the object.
        object_path (str): The path to the object.

    Returns:
        wandb.Table: The retrieved object.
    """
    alias_to_version = {
        alias: version
        for version in sorted(target_artifact['collections'][name]['versions'].keys())
        for alias in target_artifact['collections'][name]['versions'][version]['aliases']
    }
    version = alias_to_version[alias]
    return get_object_by_version(target_artifact, name, version, object_path)


def is_df_equal(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Asserts whether two DataFrames are equal in shape and values.

    Args:
        df1 (pd.DataFrame): The first DataFrame for comparison.
        df2 (pd.DataFrame): The second DataFrame for comparison.
    """
    assert df1.shape == df2.shape
    assert (df1.columns == df1.columns).all()
    assert (df2.values == df2.values).all()


def test_table_to_dataframe(target_df: pd.DataFrame):
    """
    Test the conversion of a wandb Table to a pandas DataFrame.

    Args:
        target_df (pd.DataFrame): The target DataFrame to be converted from a wandb Table.
    """
    sample_table = wandb.Table(dataframe=target_df)
    sample_df = table_to_dataframe(sample_table)
    assert isinstance(sample_df, pd.DataFrame)
    is_df_equal(sample_df, target_df)


def test_list_artifact_names(api: wandb.Api, target_artifact: dict, target_artifact_names: list):
    """
    Test listing artifact names without versions and aliases.

    Args:
        api (wandb.Api): The W&B Api instance.
        target_artifact (dict): The target artifact.
        target_artifact_names (list): The expected list of artifact names.
    """
    artifact_names = list_artifact_names(api, target_artifact['type'], with_versions=False, with_aliases=False)
    assert artifact_names == target_artifact_names


def test_list_artifact_names_with_versions(
    api: wandb.Api, target_artifact: dict, target_artifact_names_with_versions: list
):
    """
    Test listing artifact names with versions.

    Args:
        api (wandb.Api): The W&B Api instance.
        target_artifact (dict): The target artifact.
        target_artifact_names_with_versions (list): The expected list of artifact names.
    """
    artifact_names = list_artifact_names(api, target_artifact['type'], with_versions=True, with_aliases=False)
    assert artifact_names == target_artifact_names_with_versions


def test_list_artifact_names_with_aliases(
    api: wandb.Api, target_artifact: dict, target_artifact_names_with_aliases: list
):
    """
    Test listing artifact names with aliases.

    Args:
        api (wandb.Api): The W&B Api instance.
        target_artifact (dict): The target artifact.
        target_artifact_names_with_aliases (list): The expected list of artifact names.
    """
    artifact_names = list_artifact_names(api, target_artifact['type'], with_versions=False, with_aliases=True)
    assert artifact_names == target_artifact_names_with_aliases


def test_list_artifact_names_with_versions_and_aliases(
    api: wandb.Api, target_artifact: dict, target_artifact_names_with_versions_and_aliases: list
):
    """
    Test listing artifact names with both versions and aliases.

    Args:
        api (wandb.Api): The W&B Api instance.
        target_artifact (dict): The target artifact.
        target_artifact_names_with_versions_and_aliases (list): The expected list of artifact names.
    """
    artifact_names = list_artifact_names(api, target_artifact['type'], with_versions=True, with_aliases=True)
    assert artifact_names == target_artifact_names_with_versions_and_aliases


def test_load_artifact_with_versions(api: wandb.Api, target_artifact: dict, target_artifact_names_with_versions: list):
    """
    Test loading artifacts with versions.

    Args:
        api (wandb.Api): The W&B Api instance.
        target_artifact (dict): The target artifact.
        target_artifact_names_with_versions (list): The list of artifact names with versions.
    """
    for name_with_version in target_artifact_names_with_versions:
        name, version = extract_name_and_alias(name_with_version)
        target_table = get_object_by_version(target_artifact, name, version, TEST_TABLE_NAME)

        loaded_artifact = load_artifact(api, target_artifact['type'], name, version)
        assert loaded_artifact is not None

        loaded_table = loaded_artifact.get(TEST_TABLE_NAME)
        assert target_table == loaded_table


def test_load_artifact_with_aliases(api: wandb.Api, target_artifact: dict, target_artifact_names_with_aliases: list):
    """
    Test loading artifacts with aliases.

    Args:
        api (wandb.Api): The W&B Api instance.
        target_artifact (dict): The target artifact.
        target_artifact_names_with_aliases (list): The list of artifact names with aliases.
    """
    for name_with_alias in target_artifact_names_with_aliases:
        name, alias = extract_name_and_alias(name_with_alias)
        target_table = get_object_by_alias(target_artifact, name, alias, TEST_TABLE_NAME)

        loaded_artifact = load_artifact(api, target_artifact['type'], name, alias)
        assert loaded_artifact is not None

        loaded_table = loaded_artifact.get(TEST_TABLE_NAME)
        assert target_table == loaded_table


def test_load_artifact_with_wrong_type(api: wandb.Api):
    """
    Test loading an artifact with a wrong type.

    Args:
        api (wandb.Api): The W&B Api instance.
    """
    available_artifact_types = [t.name for t in api.artifact_types()]
    artifact_names = list_artifact_names(api, available_artifact_types[0], with_versions=False, with_aliases=False)
    artifact_name = artifact_names[0]
    wrong_artifact_type = 'test-wrong-type-' + '-'.join(available_artifact_types)
    artifact = load_artifact(api, wrong_artifact_type, artifact_name, 'latest')
    assert artifact is None


def test_load_artifact_with_wrong_name(api: wandb.Api):
    """
    Test loading an artifact with a wrong name.

    Args:
        api (wandb.Api): The W&B Api instance.
    """
    artifact_type = api.artifact_types()[0].name
    artifact_names = list_artifact_names(api, artifact_type, with_versions=False, with_aliases=False)
    wrong_artifact_name = 'test-wrong-name-' + '-'.join(artifact_names)
    artifact = load_artifact(api, artifact_type, wrong_artifact_name, 'latest')
    assert artifact is None


def test_load_artifact_with_wrong_alias(api: wandb.Api):
    """
    Test loading an artifact with a wrong alias.

    Args:
        api (wandb.Api): The W&B Api instance.
    """
    artifact_type = api.artifact_types()[0].name
    collection = api.artifact_type(artifact_type).collections()[0]
    artifact_name = collection.name
    aliases = collection.aliases
    wrong_alias = 'test-wrong-alias-' + '-'.join(aliases)
    artifact = load_artifact(api, artifact_type, artifact_name, wrong_alias)
    assert artifact is None
