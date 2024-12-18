from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.logging_config import setup_logger
from ..utils.other import save_chart

logger = setup_logger(__name__)


class NumericalVisualizer:
    """
    Contains methods that generate eda charts for numerical data. Will
    be fed with just numerical columns from original dataset. All methods
    will be called in order defined in :obj:`order`. Each method that would
    be called should return a tuple of (path_to_chart, chart title for latex) -
    if there is no need for chart generation should return ("", "").
    Charts should be saved via :obj:`save_chart`.
    """

    order = [
        "correlation_chart",
    ]

    @staticmethod
    def correlation_chart(df: pd.DataFrame) -> Tuple[str, str]:
        """
        Generates a heatmap of the correlation matrix for the given DataFrame.
        """

        try:
            logger.start_operation(
                f"Correlation matrix generation for {len(df.columns)} features."
            )
            plt.figure(figsize=(12, 8))

            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            plt.title("Feature Correlation Matrix")

            path = save_chart(name="correlation_matrix.png")
            logger.end_operation()

            return path, "Correlation matrix."
        except Exception as e:
            logger.error(f"Failed to generate correlation matrix: {str(e)}")
            raise
