from typing import Tuple

import matplotlib.pyplot as plt  # noqa: F401
import pandas as pd
import seaborn as sns  # noqa: F401

from ..utils.logging_config import setup_logger
from ..utils.other import save_chart  # noqa: F401

logger = setup_logger(__name__)


class CategoricalVisualizer:
    """
    Contains methods that generate eda charts for categorical data. Will
    be fed with just categorical columns from original dataset. All methods
    will be called in order defined in :obj:`order`. Each method that would
    be called should return a tuple of (path_to_chart, chart title for latex) -
    if there is no need for chart generation should return ("", "").
    Charts should be saved via :obj:`save_chart`.
    """

    order = [
        "my_chart",
    ]

    @staticmethod
    def my_chart(df: pd.DataFrame) -> Tuple[str, str]:
        pass
