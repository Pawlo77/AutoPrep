import copy
import itertools
import json
from time import time
from typing import Dict, List

import humanize
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ..raporting.raport import Report
from ..utils.abstract import ModulesHandler, Step
from ..utils.config import config
from ..utils.logging_config import setup_logger
from ..utils.other import save_model

logger = setup_logger(__name__)


class PreprocessingHandler(ModulesHandler):
    def __init__(self):
        self._pipeline_steps: List[List[Step]] = []
        self._pipeline_steps_exploded: List[List[Step]] = []
        self._pipelines: List[Pipeline] = []
        self._fit_durations: List[float] = []
        self._score_durations: List[float] = []
        self._fit_time: float = None
        self._score_time: float = None
        self._pipelines_scores: pd.Series = []
        self._best_pipelines_idx: List[int] = []
        self._model = None
        self._score_func = None

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        task: str,
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
            task (str): classification or regression.
        """

        logger.start_operation("Preprocessing.")
        logger.info("Creating pipelines...")

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

        self._pipeline_steps_exploded
        for pipeline_steps in self._pipeline_steps:
            current_pipeline_steps_exploded = PreprocessingHandler._explode_steps(
                pipeline_steps
            )
            logger.debug(
                f"Exploaded {len(pipeline_steps)} steps into {len(current_pipeline_steps_exploded)} steps."
            )
            for entry in current_pipeline_steps_exploded:
                self._pipelines.append(Pipeline(entry))
                self._pipeline_steps_exploded.append(entry)

        t0 = time()
        logger.info("Fitting pipelines...")
        for pipeline in self._pipelines:
            t1 = time()
            pipeline.fit(X_train, y_train)
            self._fit_durations(time() - t1)
        self._fit_time = time() - t0

        logger.info("Scoring pipelines...")
        t0 = time()
        self._model = (
            config.regression_pipeline_scoring_model
            if task == "regression"
            else config.classification_pipeline_scoring_model
        )
        self._score_func = (
            config.regression_pipeline_scoring_func
            if task == "regression"
            else config.classification_pipeline_scoring_func
        )
        for pipeline in self._pipelines:
            t1 = time()
            self._pipelines_scores.append(
                PreprocessingHandler.score_pipeline_with_model(
                    pipeline=pipeline,
                    model=copy.deepcopy(self._model),
                    score_func=self._score_func,
                    X_val=X_valid,
                    y_val=y_valid,
                )
            )
            self._score_durations(time() - t1)
        self._pipelines_scores = pd.Series(self._pipelines_scores)

        self._best_pipelines_idx = (
            self._pipelines_scores.nlargest(config.max_datasets_after_preprocessing)
            .sort_values(ascending=False)
            .index
        )

        for score_idx, idx in enumerate(self._best_pipelines_idx):
            save_model(
                f"preprocessing_pipeline_{score_idx}.joblib", self._pipelines[idx]
            )
        self._pipelines = []  # to save space

        self._score_time = time() - t0

        logger.end_operation()

    def write_to_raport(self, raport: Report):
        """Writes overview section to a raport"""

        preprocessing_section = raport.add_section("Preprocessing")  # noqa: F841

        pipeline_scores_description = self._pipelines_scores.describe().to_dict()
        prefixed_pipeline_scores_description = {
            f"scores_{key}": value for key, value in pipeline_scores_description.items()
        }
        statistics = {
            "Unique created pipelines": len(self._pipeline_steps),
            "All created pipelines (after exploading each step params)": len(
                self._pipeline_steps_exploded
            ),
            "All pipelines fit time": humanize.naturaltime(self._fit_time),
            "All pipelines score time": humanize.naturaltime(self._score_time),
            **prefixed_pipeline_scores_description,
            "Scoring function": self._score_func.__name__,
            "Scoring model": self._model.__name__,
        }

        raport.add_table(
            statistics,
            caption="Preprocessing pipelines runtime statistics.",
        )

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

        best_pipelines_overview = {}
        for score_idx, idx in enumerate(self._best_pipelines_idx):
            best_pipelines_overview[score_idx] = [
                f"preprocessing_pipeline_{score_idx}.joblib",
                self._pipelines_scores[idx],
                self._fit_durations[idx],
                self._score_durations[idx],
            ]
        raport.add_table(
            best_pipelines_overview,
            caption="Best preprocessing pipelines.",
            header=[
                "score index",
                "file name",
                "score",
                "fit duration",
                "score duration",
            ],
        )

        for score_idx, idx in enumerate(self._best_pipelines_idx):
            pipeline_steps_overview = {}
            for i, step in enumerate(self._pipeline_steps_exploded[idx]):
                tex = step.to_tex()
                pipeline_steps_overview[i] = [
                    step.__name__,
                    tex.pop("desc", "Yet another step."),
                    json.dumps(tex.pop("params", {})),
                ]

            raport.add_table(
                pipeline_steps_overview,
                caption=f"{score_idx}th best pipeline overwiev.",
                header=[
                    "step",
                    "description",
                    "params",
                ],
            )

        return raport

    @staticmethod
    def score_pipeline_with_model(
        preprocessing_pipeline: Pipeline,
        model: BaseEstimator,
        score_func: callable,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """
        Evaluates the performance of a given preprocessing pipeline with a model on validation data.

        Args:
            preprocessing_pipeline (Pipeline): The preprocessing pipeline to be evaluated.
            model (BaseEstimator): The model to be used for scoring.
            score_func (callable): scoring function for model predictions and y_val.
            X_val (pd.DataFrame): The validation features.
            y_val (pd.Series): The validation labels.

        Returns:
            float: The score of the pipeline on the validation data.
        """
        full_pipeline = Pipeline(
            [("preprocessing", preprocessing_pipeline), ("model", model)]
        )

        full_pipeline.fit(X_val, y_val)
        y_pred = full_pipeline.predict(X_val)

        score = score_func(y_val, y_pred)
        return score

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
