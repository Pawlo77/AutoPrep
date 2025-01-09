from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.config import config
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

    order = [
        "target_distribution_chart",
        "missing_values_chart",
    ]
    @staticmethod
    def target_distribution_chart(
        X: pd.DataFrame,  # noqa: F841
        y: pd.Series,
        task: str = "classification",
    ) -> Tuple[str, str]:
        """
        Generates a plot to visualize the distribution of the target variable.
        
        Args:
            X (pd.DataFrame): Input features (not used directly, included for API consistency).
            y (pd.Series): Target variable to visualize.
            task (str): Type of task, either "classification" or "regression".
            
        Returns:
            Tuple[str, str]: Path to the saved chart and a description of the chart.
        """
        logger.start_operation("Target distribution visualization.")
        try:
            sns.set_theme(
                style=config.chart_settings["theme"],
                palette=config.chart_settings["palette"],
            )
            y_df = y.to_frame(name="target")
            plt.figure(figsize=(10, 6))
            
            if task == "classification":
                sns.countplot(
                    data=y_df,
                    x="target",
                    palette=config.chart_settings["palette"],
                    hue="target",
                )
                total = len(y)
                for p in plt.gca().patches:
                    height = p.get_height()
                    plt.gca().text(
                        p.get_x() + p.get_width() / 2,
                        height + 3,
                        f"{height / total:.2%}",
                        ha="center",
                    )
                plt.title(f"Distribution of {y.name}")
                plt.xlabel("Target Classes")
                plt.ylabel("Count")
                path = save_chart(name="target_distribution_classification.png")
                description = "Target distribution."
            
            elif task == "regression":
                sns.histplot(
                    data=y_df,
                    x="target",
                    bins=30,  
                    stat="density",
                )

                mean_value = y.mean()
                median_value = y.median()
                plt.axvline(mean_value, color="red", linestyle="--", label=f"Mean: {mean_value:.2f}")
                plt.axvline(median_value, color="green", linestyle="--", label=f"Median: {median_value:.2f}")
                
                plt.legend()
                plt.title(f"Distribution of {y.name}")
                plt.xlabel("Target Value")
                plt.ylabel("Density")
                path = save_chart(name="target_distribution_regression.png")
                description = "Target distribution."
            
            else:
                raise ValueError(f"Unsupported task type: {task}. Use 'classification' or 'regression'.")
            
            return path, description
        
        except Exception as e:
            logger.error(f"Failed to generate target distribution plot for {task}: {str(e)}")
            raise e
        
        finally:
            logger.end_operation()


    @staticmethod
    def missing_values_chart(
        X: pd.DataFrame,
        y: pd.Series,  # noqa: F841
    ) -> Tuple[str, str]:
        """
        Generates a plot to visualize the percentage of missing values for each
        feature in the given DataFrame.
        """
        logger.start_operation("Missing values visualizations.")

        try:
            plt.figure(figsize=(10, 6))
            missing = X.isnull().sum() / len(X) * 100
            missing = missing[missing > 0].sort_values(ascending=False)

            if missing.empty:
                logger.debug("No missing values found in the dataset.")
                return "", ""
            logger.debug(
                f"Will create missing values chart for {list(missing.index)} columns."
            )

            sns.barplot(
                x=missing.index,
                y=missing.values,
                palette=config.chart_settings["palette"],
            )
            plt.xticks(rotation=45)
            plt.title("Percentage of Missing Values by Feature")
            path = save_chart(name="missing_values.png")

            return path, "Missing values."
        except Exception as e:
            logger.error(f"Failed to generate missing values plot: {str(e)}")
            raise e
        finally:
            logger.end_operation()
