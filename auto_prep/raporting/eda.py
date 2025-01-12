from typing import Dict, List, Tuple

import pandas as pd

from ..utils.logging_config import setup_logger
from ..visualization.categorical import CategoricalVisualizer
from ..visualization.eda import EdaVisualizer
from ..visualization.numerical import NumericalVisualizer
from .raport import Raport

logger = setup_logger(__name__)


class EdaRaport:
    visualizers: list = [EdaVisualizer, CategoricalVisualizer, NumericalVisualizer]

    def __init__(self):
        self.charts_dt: Dict[str, List[Tuple[str, str]]] = {}

    def run(self, X: pd.DataFrame, y: pd.Series, task: str):
        """Performs dataset EDA analysis based on the given task (classification or regression)."""

        logger.start_operation("EDA.")

        try:
            for visualizer_cls in EdaRaport.visualizers:
                logger.start_operation(f"{visualizer_cls.__name__} plot generation.")
                logger.debug(
                    f"Will call plots in the following order: {visualizer_cls.order}"
                )
                self.charts_dt[visualizer_cls.__name__] = []

                for method_name in visualizer_cls.order:
                    method = getattr(visualizer_cls, method_name)

                    try:
                        chart_dt = method(X, y, task=task)
                    except TypeError:
                        chart_dt = method(X, y)

                    if isinstance(chart_dt, list) and len(chart_dt) > 0:
                        self.charts_dt[visualizer_cls.__name__].extend(chart_dt)
                    elif isinstance(chart_dt, tuple) and chart_dt[0] != "":
                        self.charts_dt[visualizer_cls.__name__].append(chart_dt)

                logger.end_operation()

        except Exception as e:
            logger.error(f"Failed to perform EDA analysis: {str(e)}")
            raise e
        finally:
            logger.end_operation()

    def write_to_raport(self, raport: Raport):
        """Writes eda section to a raport"""

        eda_section = raport.add_section("Eda")  # noqa: F841

        section_desc = "This part of the report provides basic insides to the data and the informations it holds.."
        raport.add_text(section_desc)

        for visualizer_name, charts_dt in self.charts_dt.items():
            # raport.add_subsection(visualizer_name[: -len("Visualizer")])
            if visualizer_name == "EdaVisualizer":
                raport.add_subsection("Target variable and missing values")
            elif visualizer_name == "CategoricalVisualizer":
                if charts_dt:
                    raport.add_subsection("EDA for categorical features")
            elif visualizer_name == "NumericalVisualizer":
                if charts_dt:
                    raport.add_subsection("EDA for numerical features")

            for path, caption in charts_dt:
                if caption == "Target distribution.":
                    raport.add_reference(label=caption, prefix="Figure", add_space=True)
                    raport.add_text(" shows the distribution of the target variable.")
                elif caption == "Missing values.":
                    raport.add_reference(label=caption, prefix="Figure", add_space=True)
                    raport.add_text(
                        " shows the distribution of missing values in the dataset."
                    )
                elif caption == "Numerical Features Distribution - Page 1":
                    raport.add_text(
                        "The distribution of numerical features is presented on histogram(s) below."
                    )
                elif caption == "Categorical Features Distribution - Page 1":
                    raport.add_text(
                        "The distribution of categorical features is presented on barplot(s) below."
                    )
                elif caption == "Correlation heatmap.":
                    raport.add_reference(label=caption, prefix="Figure", add_space=True)
                    raport.add_text(" shows the correlation between features.")
                elif caption == "Boxplot page 1":
                    raport.add_text(
                        "The boxplot of numerical features is presented on chart(s) below."
                    )
                raport.add_figure(path=path, caption=caption, label=caption)

        return raport
