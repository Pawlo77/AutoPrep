from typing import List, Tuple

import matplotlib.pyplot as plt  # noqa: F401
import pandas as pd
import seaborn as sns  # noqa: F401

from ..utils.config import config
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

    order = ["categorical_distribution_chart"]

    @staticmethod
    def categorical_distribution_chart(
        X: pd.DataFrame, y: pd.Series
    ) -> List[Tuple[str, str]]:
        """
        Generates a plot to visualize the distribution of categorical features.
        """
        settings = config.chart_settings
        sns.set_theme(style=settings["theme"])

        logger.start_operation("Categorical distribution visualization.")

        try:
            categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
            if not categorical_columns:
                logger.debug("No categorical features found in the dataset.")
                return []

            categorical_columns = [
                col for col in categorical_columns if X[col].nunique() <= 15
            ]
            if not categorical_columns:
                logger.debug(
                    "No categorical features with less than or equal to 15 unique values found in the dataset."
                )
                return []

            logger.debug(
                "Will create categorical distribution visualization chart"
                f"for {categorical_columns} columns."
            )

            max_plots_per_page = 10
            categorical_groups = [
                categorical_columns[i : i + max_plots_per_page]
                for i in range(0, len(categorical_columns), max_plots_per_page)
            ]

            chart_list = []
            for group_idx, group in enumerate(categorical_groups):
                num_columns = len(group)
                num_rows = (num_columns + 1) // 2

                fig, axes = plt.subplots(
                    num_rows,
                    2,
                    figsize=(
                        settings["plot_width"],
                        settings["plot_height_per_row"] * num_rows,
                    ),
                )
                axes = axes.flatten()

                plot_count = 0
                for column in group:
                    sns.countplot(
                        data=X,
                        y=column,
                        order=X[column].value_counts().index,
                        ax=axes[plot_count],
                        palette=sns.color_palette(
                            settings["palette"], X[column].nunique()
                        ),
                        legend=False,
                    )
                    axes[plot_count].set_title(
                        f"Distribution of {column}",
                        fontsize=settings["title_fontsize"],
                        fontweight=settings["title_fontweight"],
                    )
                    axes[plot_count].set_xlabel(
                        "Count", fontsize=settings["xlabel_fontsize"]
                    )
                    axes[plot_count].set_ylabel(
                        column, fontsize=settings["ylabel_fontsize"]
                    )

                    plot_count += 1

                for j in range(plot_count, len(axes)):
                    axes[j].axis("off")

                plt.suptitle(
                    f"Categorical Features Distribution - Page {group_idx + 1}",
                    fontsize=settings["title_fontsize"],
                    fontweight=settings["title_fontweight"],
                    y=1.0,
                )
                plt.tight_layout(pad=2.0)

                path = save_chart(
                    name=f"categorical_distribution_page_{group_idx + 1}.png"
                )
                chart_list.append(
                    (path, f"Categorical Features Distribution - Page {group_idx + 1}")
                )
        except Exception as e:
            logger.error(f"Error in categorical_distribution_chart: {e}")
            raise e
        finally:
            logger.end_operation()
        return chart_list
