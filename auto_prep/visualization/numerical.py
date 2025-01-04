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

    palette = ["#FF204E"]
    order = [
        "correlation_chart",
    ]

    @staticmethod
    def numerical_distribution_chart(X: pd.DataFrame) -> Tuple[str, str]:
        """
        Generates a plot to visualize the distribution of numerical features.
        """
        try:

            numerical_columns = [col for col in X.columns if X[col].dtype != "object"]
            num_columns = len(numerical_columns)
            num_rows = (num_columns + 1) // 2

            logger.start_operation(
                f"Numerical distribution visualization for {num_columns} features."
            )

            if numerical_columns == []:
                logger.info("No numerical features found in the dataset.")
                logger.end_operation()
                return "", ""

            fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows))
            axes = axes.flatten()

            for i, column in enumerate(numerical_columns):
                sns.histplot(
                    data=X, x=column, ax=axes[i], color=NumericalVisualizer.palette[0]
                )
                axes[i].set_title(f"Distribution of {column}")
                axes[i].set_xlabel(column)
                axes[i].set_ylabel("Count")

            axes[-1].axis("off")

            plt.tight_layout()
            path = save_chart(name="numerical_distribution.png")
            logger.end_operation()
            return path, "Numerical distribution."
        except Exception as e:
            logger.error(f"Failed to generate numerical distribution plot: {str(e)}")
            raise

    @staticmethod
    def correlation_heatmap_chart(df: pd.DataFrame) -> Tuple[str, str]:
        """
        Generates a plot to visualize the correlation between features.
        """
        try:
            numerical_columns = [col for col in df.columns if df[col].dtype != "object"]
            logger.start_operation("Correlation heatmap visualization.")

            if numerical_columns == []:
                logger.info("No numerical features found in the dataset.")
                logger.end_operation()
                return "", ""

            plt.figure(figsize=(15, 10))
            sns.heatmap(
                df[numerical_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f"
            )
            plt.title("Correlation Heatmap")
            path = save_chart(name="correlation_heatmap.png")

            logger.end_operation()
            return path, "Correlation heatmap."
        except Exception as e:
            logger.error(f"Failed to generate correlation heatmap plot: {str(e)}")
            raise

    @staticmethod
    def boxplot(X: pd.DataFrame) -> Tuple[str, str]:
        numerical_columns = [col for col in X.columns if X[col].dtype != "object"]
        num_columns = len(numerical_columns)
        num_rows = (num_columns + 1) // 2

        logger.start_operation(f"Boxplot visualization for {num_columns} features.")

        if not numerical_columns:
            logger.info("No numerical features found in the dataset.")
            logger.end_operation()
            return "", ""

        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows))
        axes = axes.flatten()

        for i, column in enumerate(numerical_columns):
            sns.boxplot(
                data=X, x=column, ax=axes[i], color=NumericalVisualizer.palette[0]
            )
            axes[i].set_title(f"Boxplot of {column}")
            axes[i].set_xlabel(column)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        path = save_chart(name="boxplot.png")
        logger.end_operation()
        return path, "Boxplot."
