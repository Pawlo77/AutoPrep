import itertools
from typing import Dict, List

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
            pipeline_steps_exploded = PreprocessingHandler._explode_steps(
                pipeline_steps
            )
            logger.debug(
                f"Exploaded {len(pipeline_steps)} steps into {len(pipeline_steps_exploded)} steps."
            )
            for entry in pipeline_steps_exploded:
                self._pipelines.append(
                    Pipeline(
                        entry,
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

    @staticmethod
    def _explode_steps(steps: List[Step]) -> List[List[Step]]:
        """
        For each step with class attribute PARAMS_GRID it exploades the grid.
        """
        exploaded_steps = []
        for step in steps:
            grid = getattr(step, "PARAMS_GRID", None)

            logger.debug(
                f"Exploding for step {step.__name__}. Begining with {len(exploaded_steps)}"
            )

            steps_to_add = []
            if grid is not None:
                all_possibilities = PreprocessingHandler._exploade_grid(grid)
                steps_to_add = [
                    step(**possibility) for possibility in all_possibilities
                ]
            else:
                try:
                    steps_to_add = [step()]
                except Exception as e:
                    if "missing" in str(e) and "required positional argument" in str(e):
                        raise Exception(
                            f"{step.__name__} has no PARAM_GRID defined yet it requires params."
                        ) from e
                    raise e

            if len(exploaded_steps) == 0:
                exploaded_steps = [[step] for step in steps_to_add]
            else:
                new_exploaded_steps = []
                for step in exploaded_steps:
                    for step_to_add in steps_to_add:
                        new_exploaded_steps.append([*step, step_to_add])
                exploaded_steps = new_exploaded_steps

        return exploaded_steps

    @staticmethod
    def _exploade_grid(grid: Dict[str, List]) -> List[dict]:
        """
        Exploades dict of Lists into all possible combinations.
        """
        combinations = list(itertools.product(*grid.values()))
        return [dict(zip(grid.keys(), combination)) for combination in combinations]
