from typing import Collection, Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import BaseCrossValidator, GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


def join_cols(df: pd.DataFrame, cols: Collection[str], sep: str = "_") -> pd.Series:
    """
    Concatenate the specified columns of a DataFrame with a separator.

    Args:
        df (pd.DataFrame): The DataFrame to operate on.
        cols (Collection[str]): Column names to concatenate.
        sep (str, optional): The separator to use between the column values. Defaults to "_".

    Returns:
        pd.Series: A Series containing the concatenated values.
    """
    if len(cols) == 0:
        raise ValueError("At least a column name is required, got empthy")
    return df[cols].apply(lambda row: sep.join(row.astype(str)), axis=1)


def get_splitter(
    stratify_cols: Optional[Collection[str]] = None,
    group_cols: Optional[Collection[str]] = None,
    n_splits: int = 5,
    random_state: int = 1414,
) -> BaseCrossValidator:
    """
    Get a cross-validation splitter based on input parameters.

    Args:
        stratify_cols (Collection[str], optional): Column names for stratification. Defaults to None.
        group_cols (Collection[str], optional): Column names for grouping. Defaults to None.
        n_splits (int, optional): Number of splits in the cross-validation. Defaults to 5.
        random_state (int): Seed for random number generator. Defaults to 1414.

    Returns:
        BaseCrossValidator: A cross-validation splitter based on the input parameters.
    """
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1")

    stratify_cols = stratify_cols if stratify_cols else None
    group_cols = group_cols if group_cols else None

    unique_stratify_cols = set(stratify_cols) if stratify_cols else set()
    unique_group_cols = set(group_cols) if group_cols else set()

    if unique_stratify_cols.intersection(unique_group_cols):
        raise ValueError("group_cols and stratify_cols must be disjoint")

    if (stratify_cols is not None) and (group_cols is not None):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if stratify_cols is not None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if group_cols is not None:
        return GroupKFold(n_splits=n_splits)

    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def stratified_group_split(
    dataset_df: pd.DataFrame,
    test_fraction: Union[float, int] = 0.2,
    stratify_cols: Optional[Collection[str]] = None,
    group_cols: Optional[Collection[str]] = None,
    random_seed: int = 1414,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into development and test sets with stratification and grouping.

    Args:
        dataset_df (pd.DataFrame): The input DataFrame to be split.
        test_fraction (Union[float, int], optional): The fraction of data to be used for testing.
                                                     If a float (0, 1) is given, it's rounded to the nearest fraction.
                                                     If an integer (n > 1) is given, the fraction is calculated as 1/n.
                                                     Defaults to 0.2.
        stratify_cols (Collection[str], optional): Column names for stratification. Defaults to None.
        group_cols (Collection[str], optional): Column names for grouping. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 1414.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the development and test DataFrames.
    """
    if test_fraction <= 0:
        raise ValueError("test_fraction must be greater than 0")

    if test_fraction == 1:
        raise ValueError("test_fraction must not equal to 1")

    if isinstance(test_fraction, float) and (test_fraction > 1):
        raise ValueError("test_fraction provided as float must be less than 1")

    if isinstance(test_fraction, int):
        test_fraction = 1 / test_fraction

    split_fraction = min(test_fraction, 1 - test_fraction)
    n_splits = round(1 / split_fraction)
    splitter = get_splitter(stratify_cols, group_cols, n_splits, random_seed)

    stratify = join_cols(dataset_df, stratify_cols) if stratify_cols else None
    groups = join_cols(dataset_df, group_cols) if group_cols else None

    lowest_diff = float('inf')
    best_dev_rows, best_test_rows = None, None
    for dev_rows, test_rows in splitter.split(X=dataset_df, y=stratify, groups=groups):
        fraction = len(test_rows) / len(dataset_df)
        diff = abs(split_fraction - fraction)
        if diff < lowest_diff:
            lowest_diff = diff
            best_dev_rows = dev_rows
            best_test_rows = test_rows

    if test_fraction > 0.5:
        best_dev_rows, best_test_rows = best_test_rows, best_dev_rows

    dev_dataset_df = dataset_df.iloc[best_dev_rows].reset_index(drop=True)
    test_dataset_df = dataset_df.iloc[best_test_rows].reset_index(drop=True)

    return dev_dataset_df, test_dataset_df


def split_dataset(
    dataset_df: pd.DataFrame,
    test_fraction: Union[float, int] = 0.2,
    val_n_splits: int = 5,
    stratify_cols: Optional[Collection[str]] = None,
    group_cols: Optional[Collection[str]] = None,
    test_split_name: str = "test",
    dev_split_name: str = "dev",
    train_split_name_format: str = "train_fold_{}",
    val_split_name_format: str = "val_fold_{}",
    random_seed: int = 1414,
) -> Dict[str, pd.DataFrame]:
    """
    Split a dataset into development, test, and cross-validation sets with stratification and grouping.

    The dataset will be split into a development set and a test set. The development set will then be further
    split into k-fold cross-validation sets, each containing its own training and validation sets.
    The final data splits include a test set, k training sets, and k validation sets.

    Args:
        dataset_df (pd.DataFrame): The input DataFrame to be split.
        test_fraction (Union[float, int], optional): The fraction of data to be used for testing.
                                                     If a float is given, it's rounded to the nearest fraction.
                                                     If an integer (n) is given, the fraction is calculated as 1/n.
                                                     Defaults to 0.2.
        val_n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        stratify_cols (Collection[str], optional): Column names for stratification. Defaults to None.
        group_cols (Collection[str], optional): Column names for grouping. Defaults to None.
        test_split_name (str, optional): Name for the test split. Defaults to "test".
        dev_split_name (str, optional): Name for the development split. Defaults to "dev".
        train_split_name_format (str, optional): Format for naming training splits. Defaults to "train_fold_{}".
        val_split_name_format (str, optional): Format for naming validation splits. Defaults to "val_fold_{}".
        random_seed (int, optional): Random seed for reproducibility. Defaults to 1414.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the split DataFrames.
    """
    if val_n_splits <= 1:
        raise ValueError("val_n_splits must be greater than 1")

    data_splits = dict()

    # split into dev and test datasets
    dev_dataset_df, test_dataset_df = stratified_group_split(
        dataset_df, test_fraction, stratify_cols, group_cols, random_seed
    )
    data_splits[dev_split_name] = dev_dataset_df
    data_splits[test_split_name] = test_dataset_df

    # cross-validation split
    k_fold_splitter = get_splitter(stratify_cols, group_cols, val_n_splits, random_seed)

    dev_stratify = join_cols(dev_dataset_df, stratify_cols) if stratify_cols else None
    dev_groups = join_cols(dev_dataset_df, group_cols) if group_cols else None

    for n, (train_rows, val_rows) in enumerate(
        k_fold_splitter.split(X=dev_dataset_df, y=dev_stratify, groups=dev_groups)
    ):
        k = n + 1
        data_splits[train_split_name_format.format(k)] = dev_dataset_df.iloc[train_rows].reset_index(drop=True)
        data_splits[val_split_name_format.format(k)] = dev_dataset_df.iloc[val_rows].reset_index(drop=True)

    return data_splits


def split_dataset_v2(
    dataset_df: pd.DataFrame,
    val_fraction: Union[float, int] = 0.1,
    test_n_splits: int = 5,
    stratify_cols: Optional[Collection[str]] = None,
    group_cols: Optional[Collection[str]] = None,
    train_split_name_format: str = "train_fold_{}",
    val_split_name_format: str = "val_fold_{}",
    test_split_name_format: str = "test_fold_{}",
    random_seed: int = 1414,
) -> Dict[str, pd.DataFrame]:
    """
    Split a dataset into k-fold cross-validation sets with stratification and grouping.

    The dataset will be split into k-fold cross-validation sets, each containing development and test sets.
    For each fold, the development set will be further split into training and validation sets.
    The final data splits include k test sets, k training sets, and k validation sets.

    Args:
        dataset_df (pd.DataFrame): The input DataFrame to be split.
        val_fraction (Union[float, int], optional): The fraction of data to be used for validation.
                                                     If a float is given, it's rounded to the nearest fraction.
                                                     If an integer (n) is given, the fraction is calculated as 1/n.
                                                     Defaults to 0.2.
        test_n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        stratify_cols (Collection[str], optional): Column names for stratification. Defaults to None.
        group_cols (Collection[str], optional): Column names for grouping. Defaults to None.
        train_split_name_format (str, optional): Format for naming training splits. Defaults to "train_fold_{}".
        val_split_name_format (str, optional): Format for naming validation splits. Defaults to "val_fold_{}".
        test_split_name_format (str, optional): Format for naming validation splits. Defaults to "test_fold_{}".
        random_seed (int, optional): Random seed for reproducibility. Defaults to 1414.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the split DataFrames.
    """
    if test_n_splits <= 1:
        raise ValueError("test_n_splits must be greater than 1")

    data_splits = dict()

    # cross-validation split
    k_fold_splitter = get_splitter(stratify_cols, group_cols, test_n_splits, random_seed)

    stratify = join_cols(dataset_df, stratify_cols) if stratify_cols else None
    groups = join_cols(dataset_df, group_cols) if group_cols else None

    for n, (dev_rows, test_rows) in enumerate(k_fold_splitter.split(X=dataset_df, y=stratify, groups=groups)):
        k = n + 1
        data_splits[test_split_name_format.format(k)] = dataset_df.iloc[test_rows].reset_index(drop=True)

        # split into training and validation sets
        dev_dataset_df = dataset_df.iloc[dev_rows].reset_index(drop=True)
        train_dataset_df, val_dataset_df = stratified_group_split(
            dev_dataset_df, val_fraction, stratify_cols, group_cols, random_seed
        )
        data_splits[train_split_name_format.format(k)] = train_dataset_df
        data_splits[val_split_name_format.format(k)] = val_dataset_df

    return data_splits
