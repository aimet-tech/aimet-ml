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
    return df[cols].apply(lambda row: sep.join(row.astype(str)), axis=1)


def get_splitter(
    groups: Optional[Collection[str]] = None,
    stratify: Optional[Collection[str]] = None,
    n_splits: int = 5,
    random_state: int = 1414,
) -> BaseCrossValidator:
    """
    Get a cross-validation splitter based on input parameters.

    Args:
        groups (Collection[str], optional): Group labels for each sample. Defaults to None.
        stratify (Collection[str], optional): An array-like structure for stratified sampling. Defaults to None.
        n_splits (int, optional): Number of splits in the cross-validation. Defaults to 5.
        random_state (int): Seed for random number generator. Defaults to 1414.

    Returns:
        BaseCrossValidator: A cross-validation splitter based on the input parameters.
    """

    if (groups is not None) and (stratify is not None):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if stratify is not None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if groups is not None:
        return GroupKFold(n_splits=n_splits)

    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def stratified_group_split(
    dataset_df: pd.DataFrame,
    test_size: Union[float, int] = 0.2,
    stratify_cols: Collection[str] = (),
    group_cols: Collection[str] = (),
    random_seed: int = 1414,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into development and test sets with stratification and grouping.

    Args:
        dataset_df (pd.DataFrame): The input DataFrame to be split.
        test_size (Union[float, int], optional): The fraction of data to be used for testing
                                                 or an absolute number of test samples. Defaults to 0.2.
        stratify_cols (Collection[str], optional): Column names for stratification. Defaults to ().
        group_cols (Collection[str], optional): Column names for grouping. Defaults to ().
        random_seed (int, optional): Random seed for reproducibility. Defaults to 1414.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the development and test DataFrames.
    """
    stratify = join_cols(dataset_df, stratify_cols) if len(stratify_cols) > 0 else None
    groups = join_cols(dataset_df, group_cols) if len(group_cols) > 0 else None

    if isinstance(test_size, int):
        test_size = test_size / len(dataset_df)

    splitter = get_splitter(groups, stratify, int(1 / test_size), random_seed)
    dev_rows, test_rows = next(splitter.split(X=dataset_df, y=stratify, groups=groups))

    dev_dataset_df = dataset_df.iloc[dev_rows].reset_index(drop=True)
    test_dataset_df = dataset_df.iloc[test_rows].reset_index(drop=True)

    return dev_dataset_df, test_dataset_df


def split_dataset(
    dataset_df: pd.DataFrame,
    test_size: Union[float, int] = 0.2,
    val_n_splits: int = 5,
    stratify_cols: Collection[str] = (),
    group_cols: Collection[str] = (),
    random_seed: int = 1414,
    test_split_name: str = "test",
    dev_split_name: str = "dev",
    train_split_name_format: str = "train_fold_{}",
    val_split_name_format: str = "val_fold_{}",
) -> Dict[str, pd.DataFrame]:
    """
    Split a dataset into development, test, and cross-validation sets with stratification and grouping.

    Args:
        dataset_df (pd.DataFrame): The input DataFrame to be split.
        test_size (Union[float, int], optional): The fraction of data to be used for testing
                                                 or an absolute number of test samples. Defaults to 0.2.
        val_n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        stratify_cols (Collection[str], optional): Column names for stratification. Defaults to ().
        group_cols (Collection[str], optional): Column names for grouping. Defaults to ().
        random_seed (int, optional): Random seed for reproducibility. Defaults to 1414.
        test_split_name (str, optional): Name for the test split. Defaults to "test".
        dev_split_name (str, optional): Name for the development split. Defaults to "dev".
        train_split_name_format (str, optional): Format for naming training splits. Defaults to "train_fold_{}".
        val_split_name_format (str, optional): Format for naming validation splits. Defaults to "val_fold_{}".

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the split DataFrames.
    """
    data_splits = dict()

    # split into dev and test datasets
    dev_dataset_df, test_dataset_df = stratified_group_split(
        dataset_df, test_size, stratify_cols, group_cols, random_seed
    )
    data_splits[dev_split_name] = dev_dataset_df
    data_splits[test_split_name] = test_dataset_df

    # cross-validation split
    dev_stratify = join_cols(dev_dataset_df, stratify_cols) if len(stratify_cols) > 0 else None
    dev_groups = join_cols(dev_dataset_df, group_cols) if len(group_cols) > 0 else None

    k_fold_splitter = get_splitter(dev_groups, dev_stratify, val_n_splits, random_seed)

    for n, (train_rows, val_rows) in enumerate(
        k_fold_splitter.split(X=dev_dataset_df, y=dev_stratify, groups=dev_groups)
    ):
        k = n + 1
        data_splits[train_split_name_format.format(k)] = dev_dataset_df.iloc[train_rows].reset_index(drop=True)
        data_splits[val_split_name_format.format(k)] = dev_dataset_df.iloc[val_rows].reset_index(drop=True)

    return data_splits
