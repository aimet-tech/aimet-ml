import numpy as np
import pandas as pd
import pytest

# from aimet_ml.model_selection import get_splitter, join_cols, split_dataset, stratified_group_split


@pytest.fixture
def sample_df():
    return pd.DataFrame({'group': np.random.randint(0, 5, 100), 'stratify': np.random.randint(0, 3, 100)})


def test_join_cols():
    pass
