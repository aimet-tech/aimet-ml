from typing import Any, Dict, List, Union


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary.

    Args:
        d (dict): The input dictionary to flatten.
        prefix (str, optional): The prefix to be added to flattened keys. Defaults to "".

    Returns:
        dict: A flattened dictionary.
    """
    flat_dict = {}
    for key, value in d.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, prefix=new_key))
        else:
            flat_dict[new_key] = value
    return flat_dict


def add_metric_to_report(
    cls_report: Dict[str, Dict[str, Any]],
    metric_name: str,
    label_names: List[str],
    metric_values: List[Union[float, int]],
) -> None:
    """
    Adds metric values to a classification report.

    Args:
        cls_report (dict): The classification report as a dictionary.
        metric_name (str): The name of the metric to add.
        label_names (list): List of label names.
        metric_values (list): List of metric values corresponding to label_names.

    Raises:
        AssertionError: If the lengths of label_names and metric_values do not match.
    """

    assert len(label_names) == len(metric_values)

    cls_report["macro avg"]["average_precision"] = 0
    cls_report["weighted avg"]["average_precision"] = 0

    for label_name, metric_value in zip(label_names, metric_values):
        macro_w = 1 / len(label_names)
        micro_w = cls_report[label_name]["support"] / cls_report["weighted avg"]["support"]

        cls_report[label_name][metric_name] = metric_value
        cls_report["macro avg"][metric_name] += metric_value * macro_w
        cls_report["weighted avg"][metric_name] += metric_value * micro_w
