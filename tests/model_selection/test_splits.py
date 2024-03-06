from contextlib import nullcontext as does_not_raise
from typing import Any, Collection, Optional, Union

import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from aimet_ml.model_selection import get_splitter, join_cols, split_dataset, split_dataset_v2, stratified_group_split


def validate_splits(
    test_fraction: Union[float, int],
    stratify_cols: Optional[Collection[str]],
    group_cols: Optional[Collection[str]],
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Validate the splits created by stratified_group_split and split_dataset functions.

    Args:
        test_fraction (Union[float, int], optional): The fraction of data to be used for testing.
                                                     If a float is given, it's rounded to the nearest fraction.
                                                     If an integer (n) is given, the fraction is calculated as 1/n.
        stratify_cols (Collection[str], optional): Column names for stratification.
        group_cols (Collection[str], optional): Column names for grouping.
        dev_df (pd.DataFrame): The development dataset.
        test_df (pd.DataFrame): The test dataset.
    """
    if isinstance(test_fraction, int):
        test_fraction = 1 / test_fraction
    fraction = len(test_df) / (len(dev_df) + len(test_df))
    assert (0.5 - test_fraction) * (0.5 - fraction) >= 0
    assert abs(test_fraction - fraction) < 0.1

    if stratify_cols:
        dev_joined_stratify = join_cols(dev_df, stratify_cols)
        test_joined_stratify = join_cols(test_df, stratify_cols)
        dev_sorted_unique_values = dev_joined_stratify.value_counts(sort=True).index.to_list()
        test_sorted_unique_values = test_joined_stratify.value_counts(sort=True).index.to_list()
        assert dev_sorted_unique_values == test_sorted_unique_values

    if group_cols:
        dev_joined_group = join_cols(dev_df, group_cols)
        test_joined_group = join_cols(test_df, group_cols)
        common_values = set(dev_joined_group).intersection(set(test_joined_group))
        assert len(common_values) == 0


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Fixture to provide a sample DataFrame for testing.

    Returns:
        pd.DataFrame: A sample DataFrame.
    """
    return pd.DataFrame(
        {
            'stratify': [1] * 50 + [2] * 30 + [3] * 20,
            'group': [1, 2, 3, 4, 5] * 20,
            'a': ['a'] * 100,
            'b': ['b'] * 100,
            'c': [3] * 100,
        }
    )


@pytest.mark.parametrize(
    "cols, sep, expectation",
    [
        (['a', 'b', 'c'], '-', does_not_raise('a-b-3')),
        (['a', 'b', 'c'], '.', does_not_raise('a.b.3')),
        (['a'], '.', does_not_raise('a')),
        (['c'], '.', does_not_raise('3')),
        ([], '.', pytest.raises(ValueError, match="At least a column name is required, got empthy")),
    ],
)
def test_join_cols(cols: Collection[str], sep: str, expectation: Any, sample_df: pd.DataFrame):
    """
    Test the join_cols function.

    Args:
        cols (Collection[str]): Column names to concatenate.
        sep (str): The separator to use between the column values.
        expectation (Any): The expected outcome of the test.
        sample_df (pd.DataFrame): A sample DataFrame.
    """
    with expectation:
        joined = join_cols(sample_df, cols, sep)
        assert all(joined == expectation.enter_result)


@pytest.mark.parametrize(
    "stratify_cols, group_cols, n_splits, expectation",
    [
        (['stratify'], ['group'], 3, does_not_raise(StratifiedGroupKFold)),
        (['stratify'], [], 3, does_not_raise(StratifiedKFold)),
        (['stratify'], None, 3, does_not_raise(StratifiedKFold)),
        ([], ['group'], 3, does_not_raise(GroupKFold)),
        (None, ['group'], 3, does_not_raise(GroupKFold)),
        ([], [], 3, does_not_raise(KFold)),
        (None, None, 3, does_not_raise(KFold)),
        ([], None, 3, does_not_raise(KFold)),
        (None, None, 1, pytest.raises(ValueError, match="n_splits must be greater than 1")),
        (
            ['stratify'],
            ['stratify', 'group'],
            3,
            pytest.raises(ValueError, match="group_cols and stratify_cols must be disjoint"),
        ),
    ],
)
def test_get_splitter(
    stratify_cols: Optional[Collection[str]],
    group_cols: Optional[Collection[str]],
    n_splits: int,
    expectation: Any,
):
    """
    Test the get_splitter function.

    Args:
        stratify_cols (Collection[str], optional): Column names for stratification.
        group_cols (Collection[str], optional): Column names for grouping.
        n_splits (int, optional): Number of splits in the cross-validation.
        expectation (Any): The expected outcome of the test.
    """
    with expectation:
        splitter = get_splitter(stratify_cols, group_cols, n_splits)
        assert isinstance(splitter, expectation.enter_result)
        assert splitter.n_splits == n_splits


@pytest.mark.parametrize(
    "test_fraction, stratify_cols, group_cols, expectation",
    [
        (0.2, ['stratify'], ['group'], does_not_raise()),
        (0.8, ['stratify'], ['group'], does_not_raise()),
        (5, ['stratify'], ['group'], does_not_raise()),
        (0.4, None, ['group'], does_not_raise()),
        (0.6, None, ['group'], does_not_raise()),
        (2, None, ['group'], does_not_raise()),
        (0.3, ['stratify'], None, does_not_raise()),
        (0.7, ['stratify'], None, does_not_raise()),
        (3, ['stratify'], None, does_not_raise()),
        (0.25, None, None, does_not_raise()),
        (0.75, None, None, does_not_raise()),
        (4, None, None, does_not_raise()),
        (0, None, None, pytest.raises(ValueError, match="test_fraction must be greater than 0")),
        (1, None, None, pytest.raises(ValueError, match="test_fraction must not equal to 1")),
        (1.1, None, None, pytest.raises(ValueError, match="test_fraction provided as float must be less than 1")),
    ],
)
def test_stratified_group_split(
    test_fraction: Union[float, int],
    stratify_cols: Optional[Collection[str]],
    group_cols: Optional[Collection[str]],
    expectation: Any,
    sample_df: pd.DataFrame,
):
    """
    Test the stratified_group_split function.

    Args:
        test_fraction (Union[float, int], optional): The fraction of data to be used for testing.
                                                     If a float is given, it's rounded to the nearest fraction.
                                                     If an integer (n) is given, the fraction is calculated as 1/n.
        stratify_cols (Collection[str], optional): Column names for stratification.
        group_cols (Collection[str], optional): Column names for grouping.
        expectation (Any): The expected outcome of the test.
        sample_df (pd.DataFrame): A sample DataFrame.
    """
    with expectation:
        dev_df, test_df = stratified_group_split(sample_df, test_fraction, stratify_cols, group_cols)
        validate_splits(test_fraction, stratify_cols, group_cols, dev_df, test_df)


@pytest.mark.parametrize(
    "test_fraction, val_n_splits, stratify_cols, group_cols, test_split_name, dev_split_name, \
        train_split_name_format, val_split_name_format, expectation",
    [
        (0.2, 4, ['stratify'], ['group'], 'test-split', 'dev-split', 'train-fold{}', 'val-fold{}', does_not_raise()),
        (5, 4, ['stratify'], ['group'], 'test-split', 'dev-split', 'train-fold{}', 'val-fold{}', does_not_raise()),
        (0.2, 4, ['stratify'], None, 'test-split', 'dev-split', 'train-fold{}', 'val-fold{}', does_not_raise()),
        (5, 4, ['stratify'], None, 'test-split', 'dev-split', 'train-fold{}', 'val-fold{}', does_not_raise()),
        (0.2, 4, None, ['group'], 'test-split', 'dev-split', 'train-fold{}', 'val-fold{}', does_not_raise()),
        (5, 4, None, ['group'], 'test-split', 'dev-split', 'train-fold{}', 'val-fold{}', does_not_raise()),
        (0.2, 4, None, None, 'test-split', 'dev-split', 'train-fold{}', 'val-fold{}', does_not_raise()),
        (5, 4, None, None, 'test-split', 'dev-split', 'train-fold{}', 'val-fold{}', does_not_raise()),
        (
            0.2,
            1,
            None,
            None,
            'test-split',
            'dev-split',
            'train-fold{}',
            'val-fold{}',
            pytest.raises(ValueError, match="val_n_splits must be greater than 1"),
        ),
    ],
)
def test_split_dataset(
    test_fraction: Union[float, int],
    val_n_splits: int,
    stratify_cols: Optional[Collection[str]],
    group_cols: Optional[Collection[str]],
    test_split_name: str,
    dev_split_name: str,
    train_split_name_format: str,
    val_split_name_format: str,
    expectation: Any,
    sample_df: pd.DataFrame,
):
    """
    Test the split_dataset function.

    Args:
        test_fraction (Union[float, int], optional): The fraction of data to be used for testing.
                                                     If a float is given, it's rounded to the nearest fraction.
                                                     If an integer (n) is given, the fraction is calculated as 1/n.
        val_n_splits (int, optional): Number of cross-validation splits.
        stratify_cols (Collection[str], optional): Column names for stratification.
        group_cols (Collection[str], optional): Column names for grouping.
        test_split_name (str): Name for the test split.
        dev_split_name (str): Name for the development split.
        train_split_name_format (str): Format for naming training splits.
        val_split_name_format (str): Format for naming validation splits.
        expectation (Any): The expected outcome of the test.
        sample_df (pd.DataFrame): A sample DataFrame.
    """
    with expectation:
        data_splits = split_dataset(
            sample_df,
            test_fraction,
            val_n_splits,
            stratify_cols,
            group_cols,
            test_split_name,
            dev_split_name,
            train_split_name_format,
            val_split_name_format,
        )

        assert dev_split_name in data_splits.keys()
        assert test_split_name in data_splits.keys()
        dev_df = data_splits[dev_split_name]
        test_df = data_splits[test_split_name]
        validate_splits(test_fraction, stratify_cols, group_cols, dev_df, test_df)

        for n in range(val_n_splits):
            k = n + 1
            train_split_name = train_split_name_format.format(k)
            val_split_name = val_split_name_format.format(k)
            assert train_split_name in data_splits.keys()
            assert val_split_name in data_splits.keys()
            train_df = data_splits[train_split_name]
            val_df = data_splits[val_split_name]
            validate_splits(1 / val_n_splits, stratify_cols, group_cols, train_df, val_df)


@pytest.mark.parametrize(
    "val_fraction, test_n_splits, stratify_cols, group_cols, \
        train_split_name_format, val_split_name_format, test_split_name_format, expectation",
    [
        (0.25, 5, ['stratify'], ['group'], 'train-fold{}', 'val-fold{}', 'test-fold{}', does_not_raise()),
        (4, 5, ['stratify'], ['group'], 'train-fold{}', 'val-fold{}', 'test-fold{}', does_not_raise()),
        (0.25, 5, ['stratify'], None, 'train-fold{}', 'val-fold{}', 'test-fold{}', does_not_raise()),
        (4, 5, ['stratify'], None, 'train-fold{}', 'val-fold{}', 'test-fold{}', does_not_raise()),
        (0.25, 5, None, ['group'], 'train-fold{}', 'val-fold{}', 'test-fold{}', does_not_raise()),
        (4, 5, None, ['group'], 'train-fold{}', 'val-fold{}', 'test-fold{}', does_not_raise()),
        (0.25, 5, None, None, 'train-fold{}', 'val-fold{}', 'test-fold{}', does_not_raise()),
        (4, 5, None, None, 'train-fold{}', 'val-fold{}', 'test-fold{}', does_not_raise()),
        (
            0.25,
            1,
            None,
            None,
            'train-fold{}',
            'val-fold{}',
            'test-fold{}',
            pytest.raises(ValueError, match="test_n_splits must be greater than 1"),
        ),
    ],
)
def test_split_dataset_v2(
    val_fraction: Union[float, int],
    test_n_splits: int,
    stratify_cols: Optional[Collection[str]],
    group_cols: Optional[Collection[str]],
    train_split_name_format: str,
    val_split_name_format: str,
    test_split_name_format: str,
    expectation: Any,
    sample_df: pd.DataFrame,
):
    """
    Test the split_dataset_v2 function.

    Args:
        val_fraction (Union[float, int], optional): The fraction of data to be used for validation.
                                                     If a float is given, it's rounded to the nearest fraction.
                                                     If an integer (n) is given, the fraction is calculated as 1/n.
        test_n_splits (int, optional): Number of cross-validation splits.
        stratify_cols (Collection[str], optional): Column names for stratification.
        group_cols (Collection[str], optional): Column names for grouping.
        train_split_name_format (str): Format for naming training splits.
        val_split_name_format (str): Format for naming validation splits.
        test_split_name_format (str): Format for naming test splits.
        expectation (Any): The expected outcome of the test.
        sample_df (pd.DataFrame): A sample DataFrame.
    """
    with expectation:
        data_splits = split_dataset_v2(
            sample_df,
            val_fraction,
            test_n_splits,
            stratify_cols,
            group_cols,
            train_split_name_format,
            val_split_name_format,
            test_split_name_format,
        )

        for n in range(test_n_splits):
            k = n + 1
            train_split_name = train_split_name_format.format(k)
            val_split_name = val_split_name_format.format(k)
            test_split_name = test_split_name_format.format(k)

            assert train_split_name in data_splits.keys()
            assert val_split_name in data_splits.keys()
            assert test_split_name in data_splits.keys()

            train_df = data_splits[train_split_name]
            val_df = data_splits[val_split_name]
            test_df = data_splits[test_split_name]
            dev_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

            validate_splits(1 / test_n_splits, stratify_cols, group_cols, dev_df, test_df)
            validate_splits(val_fraction, stratify_cols, group_cols, train_df, val_df)


if __name__ == "__main__":
    pytest.main()
