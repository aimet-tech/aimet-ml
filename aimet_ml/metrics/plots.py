from typing import Collection

from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def get_confusion_matrix(
    y_true: Collection, y_pred: Collection, display_labels: list, name: str = ""
) -> ConfusionMatrixDisplay:
    """
    Generate and display a confusion matrix plot.

    Args:
        y_true (Collection): True labels.
        y_pred (Collection): Predicted labels.
        display_labels (list): Labels to be displayed on the confusion matrix plot.
        name (str, optional): Name of the plot. Defaults to an empty string.

    Returns:
        ConfusionMatrixDisplay: A display object for the confusion matrix plot.
    """
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=display_labels)
    cm_display.plot()
    cm_display.ax_.set_title(name)
    return cm_display


def get_prc_display(y_true: Collection, y_score: Collection, name: str = "") -> PrecisionRecallDisplay:
    """
    Generate and display a precision-recall curve plot.

    Args:
        y_true (Collection): True labels.
        y_score (Collection): Predicted scores or probabilities.
        name (str, optional): Name of the plot. Defaults to an empty string.

    Returns:
        PrecisionRecallDisplay: A display object for the precision-recall curve plot.
    """
    pr_display = PrecisionRecallDisplay.from_predictions(y_true, y_score, name=name)
    pr_display.ax_.set_title(name)
    return pr_display


def get_roc_display(y_true: Collection, y_score: Collection, name: str = "") -> RocCurveDisplay:
    """
    Generate and display a ROC curve plot.

    Args:
        y_true (Collection): True labels.
        y_score (Collection): Predicted scores or probabilities.
        name (str, optional): Name of the plot. Defaults to an empty string.

    Returns:
        RocCurveDisplay: A display object for the ROC curve plot.
    """
    roc_display = RocCurveDisplay.from_predictions(y_true, y_score, name=name)
    roc_display.ax_.set_title(name)
    return roc_display
