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

    palette = ["#FF204E"]
    order = ["categorical_distribution_chart"]

    @staticmethod
    def categorical_distribution_chart(X: pd.DataFrame) -> Tuple[str, str]:
        """
        Generates a plot to visualize the distribution of categorical features.
        """
        try:

            categorical_columns = [col for col in X.columns if X[col].dtype == "object"]
            num_columns = len(categorical_columns)
            num_rows = (num_columns + 1) // 2

            logger.start_operation(
                f"Categorical distribution visualization for {num_columns} features."
            )

            if not categorical_columns:
                logger.info("No categorical features found in the dataset.")
                logger.end_operation()
                return "", ""

            fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows))
            axes = axes.flatten()

            for i, column in enumerate(categorical_columns):
                sns.countplot(
                    data=X,
                    y=column,
                    order=X[column].value_counts().index,
                    ax=axes[i],
                    color=CategoricalVisualizer.palette[0],
                )
                axes[i].set_title(f"Distribution of {column}")
                axes[i].set_xlabel(column)
                axes[i].set_ylabel("Count")
                axes[i].tick_params(axis="x", rotation=45)

            axes[-1].axis("off")

            plt.tight_layout()
            path = save_chart(name="categorical_distribution.png")
            logger.end_operation()
            return path, "Categorical distribution."
        except Exception as e:
            logger.error(f"Failed to generate categorical distribution plot: {str(e)}")
            raise
