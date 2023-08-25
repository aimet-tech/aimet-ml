from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

PWD = Path(__file__).parent


def add_bar_label(bar_chart: plt.bar, with_percent: bool = False, percent_digits: int = 2) -> None:
    """
    Add labels to a bar chart with optional percentage values.

    Args:
        bar_chart (plt.bar): The bar chart object.
        with_percent (bool, optional): Whether to include percentage values. Defaults to False.
        percent_digits (int, optional): Number of decimal digits for percentage values. Defaults to 2.
    """
    containers = bar_chart.containers[0]
    labels = None
    if with_percent:
        datavalues = containers.datavalues
        total = sum(datavalues)
        labels = [f"{v:,} ({v/total*100:.{percent_digits}f}%)" for v in datavalues]
    bar_chart.bar_label(containers, labels)


def plt2arr(fig: Figure, draw: bool = True) -> np.ndarray:
    """
    Convert a Matplotlib figure to a NumPy array.

    Args:
        fig (Figure): The Matplotlib figure to be converted.
        draw (bool, optional): Whether to draw the figure. Defaults to True.

    Returns:
        np.ndarray: The converted figure as a NumPy array.
    """
    if draw:
        fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w, h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr


def set_font(font_path: str) -> None:
    """
    Set the font for Matplotlib using the provided font file.

    Args:
        font_path (str): Path to the font file.
    """
    import matplotlib
    import matplotlib.font_manager

    font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
    matplotlib.font_manager.fontManager.addfont(font_path)
    matplotlib.rc("font", family=font_prop.get_name())


def set_thai_font() -> None:
    """Set the Thai font for Matplotlib using a predefined font path."""
    font_path = str(PWD.parent / "resources" / "fonts" / "thsarabunnew-webfont.ttf")
    set_font(font_path)
