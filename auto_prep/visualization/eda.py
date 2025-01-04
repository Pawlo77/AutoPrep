from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.logging_config import setup_logger
from ..utils.other import save_chart

logger = setup_logger(__name__)


class EdaVisualizer:
    """
    Contains methods that generate basic eda charts. Will
    be fed with entire original dataset. All methods
    will be called in order defined in :obj:`order`. Each method that would
    be called should return a tuple of (path_to_chart, chart title for latex) -
    if there is no need for chart generation should return ("", "").
    Charts should be saved via :obj:`save_chart`.
    """

    palette = ["#FF204E"]
    order = [
        "target_distribution_chart",
        "missing_values_chart",
    ]

    # for classification
    @staticmethod
    def target_distribution_chart(df: pd.DataFrame, y) -> Tuple[str, str]:
        """
        Generates a plot to visualize the distribution of the target variable.
        """
        try:
            logger.start_operation("Target distribution visualization.")
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=y, color=EdaVisualizer.palette[0])
            # add percent labels
            total = len(df)
            for p in plt.gca().patches:
                height = p.get_height()
                plt.gca().text(
                    p.get_x() + p.get_width() / 2,
                    height + 3,
                    f"{height / total:.2%}",
                    ha="center",
                )

            plt.title(f"Distribution of {y.name}")
            path = save_chart(name="target_distribution.png")
            logger.end_operation()
            return path, "Target distribution."
        except Exception as e:
            logger.error(f"Failed to generate target distribution plot: {str(e)}")

    @staticmethod
    def missing_values_chart(df: pd.DataFrame) -> Tuple[str, str]:
        """
        Generates a plot to visualize the percentage of missing values for each
        feature in the given DataFrame.
        """

        try:

            plt.figure(figsize=(10, 6))
            missing = df.isnull().sum() / len(df) * 100
            missing = missing[missing > 0].sort_values(ascending=False)

            logger.start_operation(
                f"Missing values visualization for {len(missing)} features."
            )

            if missing.empty:
                logger.info("No missing values found in the dataset.")
                logger.end_operation()
                return "", ""

            sns.barplot(
                x=missing.index, y=missing.values, color=EdaVisualizer.palette[0]
            )
            plt.xticks(rotation=45)
            plt.title("Percentage of Missing Values by Feature")
            path = save_chart(name="missing_values.png")

            logger.end_operation()

            return path, "Missing values."
        except Exception as e:
            logger.error(f"Failed to generate missing values plot: {str(e)}")
            raise
