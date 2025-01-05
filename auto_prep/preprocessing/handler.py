from typing import List

import pandas as pd
from sklearn.pipeline import Pipeline

from ..raporting.raport import Report
from ..utils.abstract import ModulesHandler, Step
from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class PreprocessingHandler(ModulesHandler):
    def __init__(self):
        self._pipeline_steps: List[List[Step]] = []
        self._pipelines: List[Pipeline] = []

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ):
        """
        Performs dataset preprocessing and scoring.

        Preprocessing steps are as follows:
        - imputing
        - scaling
        - encoding
        - binning
        - outlier detection
        - filtering out features (ex those with 0 variance)
        - feature selection / dimentionality reduction

        Args:
            X_train (pd.DataFrame): Training feature dataset.
            y_train (pd.Series): Training target dataset.
            X_valid (pd.DataFrame): Validation feature dataset.
            y_valid (pd.Series): Validation target dataset.
        """

        logger.start_operation("Preprocessing.")

        pipelines = []
        for step_name, package_name in [
            ("Imputting missing data.", ".imputer"),
            ("Scaling data.", ".scaler"),
            ("Encoding data.", ".encoder"),
            ("Binning data.", ".binning"),
            ("Outlier detection.", ".outlier_detector"),
        ]:
            self._pipeline_steps = ModulesHandler.construct_pipelines_steps_helper(
                step_name,
                package_name,
                __file__,
                self._pipeline_steps,
                required_only_=config.perform_only_required_,
            )

        logger.debug(f"Extracted pipelines steps: {pipelines}")

        for pipeline_steps in self._pipeline_steps:
            self._pipelines.append(
                Pipeline(
                    pipeline_steps,
                )
            )

        logger.debug("Fitting pipelines...")
        for pipeline in self._pipelines:
            pipeline.fit(X_train, y_train)

        logger.end_operation()

    def write_to_raport(self, raport: Report):
        """Writes overview section to a raport"""

        preprocessing_section = raport.add_section("Preprocessing")  # noqa: F841

        pipelines_overview = {}
        for i, pipeline_steps in enumerate(self._pipeline_steps):
            pipelines_overview[i] = [
                ", ".join(step.__name__ for step in pipeline_steps)
            ]

        raport.add_table(
            pipelines_overview,
            caption="Pipelines steps overview.",
            header=["index", "steps"],
        )

        return raport
