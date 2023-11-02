from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from typing import Any, Collection, Union

import pytest

from aimet_ml.metrics import add_metric_to_report, flatten_dict


@pytest.fixture
def sample_cls_report() -> dict:
    """
    Fixture that returns a sample classification report for testing.

    Returns:
        dict: A sample classification report.
    """
    return {
        'neg': {'precision': 1.0, 'recall': 0.7, 'f1-score': 0.8235294117647058, 'support': 10.0},
        'pos': {'precision': 0.625, 'recall': 1.0, 'f1-score': 0.7692307692307693, 'support': 5.0},
        'accuracy': 0.8,
        'macro avg': {'precision': 0.8125, 'recall': 0.85, 'f1-score': 0.7963800904977376, 'support': 15.0},
        'weighted avg': {'precision': 0.875, 'recall': 0.8, 'f1-score': 0.8054298642533937, 'support': 15.0},
    }


def test_flatten_dict(sample_cls_report: dict):
    """
    Test the flatten_dict function.

    Args:
        sample_cls_report (dict): A sample classification report for testing.
    """
    original_cls_report = deepcopy(sample_cls_report)
    flattened_cls_report = flatten_dict(sample_cls_report)

    assert isinstance(flattened_cls_report, dict)
    assert flattened_cls_report["neg_precision"] == original_cls_report["neg"]["precision"]
    assert flattened_cls_report["neg_recall"] == original_cls_report["neg"]["recall"]
    assert flattened_cls_report["neg_f1-score"] == original_cls_report["neg"]["f1-score"]
    assert flattened_cls_report["neg_support"] == original_cls_report["neg"]["support"]
    assert flattened_cls_report["pos_precision"] == original_cls_report["pos"]["precision"]
    assert flattened_cls_report["pos_recall"] == original_cls_report["pos"]["recall"]
    assert flattened_cls_report["pos_f1-score"] == original_cls_report["pos"]["f1-score"]
    assert flattened_cls_report["pos_support"] == original_cls_report["pos"]["support"]
    assert flattened_cls_report["accuracy"] == original_cls_report["accuracy"]
    assert flattened_cls_report["macro avg_precision"] == original_cls_report["macro avg"]["precision"]
    assert flattened_cls_report["macro avg_recall"] == original_cls_report["macro avg"]["recall"]
    assert flattened_cls_report["macro avg_f1-score"] == original_cls_report["macro avg"]["f1-score"]
    assert flattened_cls_report["macro avg_support"] == original_cls_report["macro avg"]["support"]
    assert flattened_cls_report["weighted avg_precision"] == original_cls_report["weighted avg"]["precision"]
    assert flattened_cls_report["weighted avg_recall"] == original_cls_report["weighted avg"]["recall"]
    assert flattened_cls_report["weighted avg_f1-score"] == original_cls_report["weighted avg"]["f1-score"]
    assert flattened_cls_report["weighted avg_support"] == original_cls_report["weighted avg"]["support"]


@pytest.mark.parametrize(
    "metric_name, label_names, metric_values, expectation",
    [
        ("custom_metric", ["neg", "pos"], [0.9, 0.85], does_not_raise({'macro avg': 0.875, 'weighted avg': 0.88333})),
        (
            "custom_metric",
            ["neg"],
            [0.9, 0.85],
            pytest.raises(ValueError, match="label_names and metric_values must have the same length"),
        ),
        (
            "custom_metric",
            ["neg", "neg"],
            [0.9, 0.85],
            pytest.raises(ValueError, match="Elements in label_names must be distinct from each other"),
        ),
        (
            "custom_metric",
            ["neg", "wrong-name"],
            [0.9, 0.85],
            pytest.raises(ValueError, match="All elements in label_names must be in the target_names of cls_report"),
        ),
    ],
)
def test_add_metric_to_report(
    metric_name: str,
    label_names: Collection[str],
    metric_values: Collection[Union[float, int]],
    expectation: Any,
    sample_cls_report: dict,
):
    """
    Test the add_metric_to_report function.

    Args:
        metric_name (str): The name of the metric to add.
        label_names (Collection[str]): Collection of label names.
        metric_values (Collection[Union[float, int]]): Collection of metric values corresponding to label_names.
        expected_macro_avg (float): The expected macro-average value after adding the custom metric.
        expectation (Any): The expected outcome of the test.
        sample_cls_report (dict): A sample classification report for testing.
    """
    original_cls_report = deepcopy(sample_cls_report)
    with expectation:
        add_metric_to_report(sample_cls_report, metric_name, label_names, metric_values)

        for label_name, metric_value in zip(label_names, metric_values):
            assert sample_cls_report[label_name][metric_name] == metric_value
            del sample_cls_report[label_name][metric_name]

        expected_macro_avg = expectation.enter_result['macro avg']
        assert sample_cls_report["macro avg"][metric_name] == pytest.approx(expected_macro_avg, abs=1e-5)
        del sample_cls_report["macro avg"][metric_name]

        expected_weighted_avg = expectation.enter_result['weighted avg']
        assert sample_cls_report["weighted avg"][metric_name] == pytest.approx(expected_weighted_avg, abs=1e-5)
        del sample_cls_report["weighted avg"][metric_name]

        assert sample_cls_report == original_cls_report


if __name__ == "__main__":
    pytest.main()
