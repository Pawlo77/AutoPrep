from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.config import config
from ..utils.logging_config import setup_logger
from ..utils.other import save_chart

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

    order = ["categorical_distribution_chart"]

    @staticmethod
    def categorical_distribution_chart(
        X: pd.DataFrame,
        y: pd.Series,  # noqa: F841
    ) -> Tuple[str, str]:
        """
        Generates a plot to visualize the distribution of categorical features.
        """

        logger.start_operation("Categorical distribution visualization.")

        try:
            categorical_columns = [col for col in X.columns if X[col].dtype == "object"]
            if not categorical_columns:
                logger.debug("No categorical features found in the dataset.")
                return "", ""
            logger.debug(
                "Will create categirical distribution visualisation chart"
                f"for {categorical_columns} columns."
            )

            num_columns = len(categorical_columns)
            num_rows = (num_columns + 1) // 2

            _, axes = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows))
            axes = axes.flatten()

            for i, column in enumerate(categorical_columns):
                sns.countplot(
                    data=X,
                    y=column,
                    order=X[column].value_counts().index,
                    ax=axes[i],
                    color=config.raport_chart_color_pallete[0],
                )
                axes[i].set_title(f"Distribution of {column}")
                axes[i].set_xlabel(column)
                axes[i].set_ylabel("Count")
                axes[i].tick_params(axis="x", rotation=45)

            axes[-1].axis("off")

            plt.tight_layout()
            path = save_chart(name="categorical_distribution.png")
            return path, "Categorical distribution."
        except Exception as e:
            logger.error(f"Failed to generate categorical distribution plot: {str(e)}")
            raise e
        finally:
            logger.end_operation()
