from typing import Any, Collection, Dict, Union


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
    label_names: Collection[str],
    metric_values: Collection[Union[float, int]],
) -> None:
    """
    Adds metric values to a classification report.

    Args:
        cls_report (dict): The classification report as a dictionary.
        metric_name (str): The name of the metric to add.
        label_names (Collection[str]): Collection of label names.
        metric_values (Collection[Union[float, int]]): Collection of metric values corresponding to label_names.

    Raises:
        AssertionError: If the lengths of label_names and metric_values do not match.
    """
    if len(label_names) != len(metric_values):
        raise ValueError('label_names and metric_values must have the same length')

    if len(set(label_names)) != len(label_names):
        raise ValueError('Elements in label_names must be distinct from each other')

    if any(label_name not in cls_report for label_name in label_names):
        raise ValueError('All elements in label_names must be in the target_names of cls_report')

    cls_report["macro avg"][metric_name] = 0
    cls_report["weighted avg"][metric_name] = 0

    for label_name, metric_value in zip(label_names, metric_values):
        macro_w = 1 / len(label_names)
        micro_w = cls_report[label_name]["support"] / cls_report["weighted avg"]["support"]

        cls_report[label_name][metric_name] = metric_value
        cls_report["macro avg"][metric_name] += metric_value * macro_w
        cls_report["weighted avg"][metric_name] += metric_value * micro_w
