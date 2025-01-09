import concurrent.futures
import copy
import itertools
import json
import logging
from functools import partial
from time import time
from typing import Dict, List

import humanize
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..utils.abstract import ModulesHandler, Step
from ..utils.config import config
from ..utils.logging_config import setup_logger
from ..utils.other import save_model

logger = setup_logger(__name__)


def _fit_pipeline(pipeline, X_train, y_train):
    t1 = time()
    try:
        pipeline = pipeline.fit(X_train, y_train)
    except Exception as e:
        raise Exception(f"Error fitting pipeline {pipeline.steps}") from e
    return pipeline, time() - t1, pipeline.transform(X_train).describe().T.reset_index()


def _score_pipeline(pipeline, X_train, y_train, X_valid, y_valid, model, score_func):
    t1 = time()
    score = PreprocessingHandler.score_pipeline_with_model(
        preprocessing_pipeline=pipeline,
        model=copy.deepcopy(model),
        score_func=score_func,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )
    duration = time() - t1
    return score, duration


class PreprocessingHandler:
    def __init__(self):
        self._pipeline_steps: List[List[Step]] = []
        self._pipeline_steps_exploded: List[List[Step]] = []
        self._pipelines: List[Pipeline] = []
        self._fit_durations: List[float] = []
        self._score_durations: List[float] = []
        self._fit_time: float = None
        self._score_time: float = None
        self._pipelines_scores: pd.Series = []
        self._pipelines_descr: List[pd.DataFrame] = []
        self._best_pipelines_idx: List[int] = []
        self._model = None
        self._score_func = None
        self._target_encoder = LabelEncoder()

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

        Args:
            X_train (pd.DataFrame): Training feature dataset.
            y_train (pd.Series): Training target dataset.
            X_valid (pd.DataFrame): Validation feature dataset.
            y_valid (pd.Series): Validation target dataset.
            task (str): regiression / classification
        """

        logger.start_operation("Preprocessing.")

        if task == "regression":
            self._model = config.regression_pipeline_scoring_model
            self._score_func = config.regression_pipeline_scoring_func
        else:
            self._model = config.classification_pipeline_scoring_model
            self._score_func = config.classification_pipeline_scoring_func
            y_train = pd.Series(self._target_encoder.fit_transform(y_train))
            y_valid = pd.Series(self._target_encoder.transform(y_valid))

        logger.info("Creating pipelines...")

        # Creating pipelines
        for step_name, package_name in [
            ("Imputting missing data.", ".imputing"),
            ("Removing redundant columns.", ".redundancy_filtering"),
            ("Encoding data.", ".encoding"),
            ("Removing zero variance columns.", ".variance_filtering"),
            ("Removing correlated features.", ".correlation_filtering"),
            ("Scaling data.", ".scaling"),
            ("Features selection.", ".feature_selecting"),
            # ("Binning data.", ".binning"),
            ("Dinemtionality reduction.", ".dimention_reducing"),
        ]:
            self._pipeline_steps = ModulesHandler.construct_pipelines_steps_helper(
                step_name,
                package_name,
                __file__,
                self._pipeline_steps,
                required_only_=config.perform_only_required_,
            )

        logger.info(f"Extracted {len(self._pipeline_steps)} pipelines.")
        logger.debug(f"Extracted pipelines: {self._pipeline_steps}")

        # Exploding pipeline steps
        for pipeline_steps in self._pipeline_steps:
            current_pipeline_steps_exploded = PreprocessingHandler._explode_steps(
                pipeline_steps
            )
            logger.debug(
                f"Exploded {len(pipeline_steps)} steps into {len(current_pipeline_steps_exploded)} steps."
            )
            for entry in current_pipeline_steps_exploded:
                self._pipelines.append(Pipeline(entry))
                self._pipeline_steps_exploded.append(entry)

        logger.info(f"Exploded into {len(self._pipelines)} pipelines.")

        # Fitting and scoring inside a method (protected by the entry point)
        t0 = time()
        logger.info("Fitting pipelines...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.max_workers
        ) as executor:
            gen = executor.map(
                partial(_fit_pipeline, X_train=X_train, y_train=y_train),
                self._pipelines,
            )
            if logger.level >= logging.INFO:
                gen = tqdm(gen, desc="Fitting pipelines", unit="pipeline")
            results = list(gen)

        self._pipelines, self._fit_durations, self._pipelines_descr = zip(*results)
        self._fit_time = time() - t0

        logger.info("Scoring pipelines...")

        t0 = time()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.max_workers
        ) as executor:
            gen = executor.map(
                partial(
                    _score_pipeline,
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    model=self._model,
                    score_func=self._score_func,
                ),
                self._pipelines,
            )
            if logger.level >= logging.INFO:
                gen = tqdm(gen, desc="Scoring pipelines", unit="pipeline")
            results = list(gen)

        self._pipelines_scores, self._score_durations = zip(*results)
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

    def write_to_raport(self, raport):
        """Writes overview section to a raport"""

        preprocessing_section = raport.add_section("Preprocessing")  # noqa: F841

        section_desc = f"This part of the report presents the results of the preprocessing process. It was configured to create up to {config.max_datasets_after_preprocessing} unique preprocessing pipelines."
        raport.add_text(section_desc)

        pipeline_scores_description = self._pipelines_scores.describe().to_dict()
        prefixed_pipeline_scores_description = {
            f"scores_{key}": value for key, value in pipeline_scores_description.items()
        }
        statistics = {
            "Unique created pipelines": len(self._pipeline_steps),
            "All created pipelines (after exploading each step params)": len(
                self._pipeline_steps_exploded
            ),
            "All pipelines fit time": humanize.naturaldelta(self._fit_time),
            "All pipelines score time": humanize.naturaldelta(self._score_time),
            **prefixed_pipeline_scores_description,
            "Scoring function": type(self._score_func.__name__),
            "Scoring model": type(self._model).__name__,
        }

        raport.add_table(
            statistics,
            caption="Preprocessing pipelines runtime statistics.",
        )

        pipelines_overview = {}
        for i, pipeline_steps in enumerate(self._pipeline_steps):
            pipelines_overview[i] = ", ".join(step.__name__ for step in pipeline_steps)

        raport.add_table(
            pipelines_overview,
            caption="Pipelines steps overview.",
            header=["index", "steps"],
            widths=[20, 160],
        )

        best_pipelines_overview = []
        for score_idx, idx in enumerate(self._best_pipelines_idx):
            best_pipelines_overview.append(
                [
                    score_idx,
                    f"preprocessing_pipeline_{score_idx}.joblib",
                    self._pipelines_scores[idx],
                    humanize.naturaldelta(self._fit_durations[idx]),
                    humanize.naturaldelta(self._score_durations[idx]),
                ]
            )
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
            pipeline_steps_overview = []
            for i, step in enumerate(self._pipeline_steps_exploded[idx]):
                tex = step[1].to_tex()
                pipeline_steps_overview.append(
                    [
                        i,
                        type(step[1]).__name__,
                        tex.pop("desc", "Yet another step."),
                        json.dumps(tex.pop("params", {})),
                    ]
                )

            raport.add_table(
                pipeline_steps_overview,
                caption=f"{score_idx}th best pipeline overwiev on training set.",
                header=[
                    "step",
                    "name",
                    "description",
                    "params",
                ],
                widths=[10, 30, 60, 60],
            )

            columns = [
                c.replace("%", "\%")  # noqa W605
                for c in self._pipelines_descr[score_idx].columns
            ]
            raport.add_table(
                self._pipelines_descr[score_idx].values.tolist(),
                caption=f"{score_idx}th best pipeline output overview.",
                header=columns,
            )

        return raport

    @staticmethod
    def score_pipeline_with_model(
        preprocessing_pipeline: Pipeline,
        model: BaseEstimator,
        score_func: callable,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> float:
        """
        Evaluates the performance of a given preprocessing pipeline with a model on validation data.

        Args:
            preprocessing_pipeline (Pipeline): The preprocessing pipeline to be evaluated.
            model (BaseEstimator): The model to be used for scoring.
            score_func (callable): scoring function for model predictions and y_val.
            X_train (pd.DataFrame): Training feature dataset.
            y_train (pd.Series): Training target dataset.
            X_valid (pd.DataFrame): Validation feature dataset.
            y_valid (pd.Series): Validation target dataset.

        Returns:
            float: The score of the pipeline on the validation data.
        """
        X_train = preprocessing_pipeline.transform(X_train)
        X_valid = preprocessing_pipeline.transform(X_valid)

        try:
            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
        except Exception as e:
            raise Exception(
                f"Scorring pipeline {preprocessing_pipeline.steps} failed."
            ) from e

        score = score_func(y_valid, y_pred)
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
                    (step.__name__, step(**possibility))
                    for possibility in all_possibilities
                ]
            else:
                try:
                    steps_to_add = [(step.__name__, step())]
                except Exception as e:
                    if "missing" in str(e) and "required positional argument" in str(e):
                        raise Exception(
                            f"{step.__name__} has no PARAM_GRID defined yet it requires params."
                        ) from e
                    raise e

            if len(exploaded_steps) == 0:
                exploaded_steps = [[entry] for entry in steps_to_add]
            else:
                new_exploaded_steps = []
                for entry in exploaded_steps:
                    for entry_to_add in steps_to_add:
                        new_exploaded_steps.append(
                            [*[copy.deepcopy(e) for e in entry], entry_to_add]
                        )
                exploaded_steps = new_exploaded_steps

        return exploaded_steps

    @staticmethod
    def _exploade_grid(grid: Dict[str, List]) -> List[dict]:
        """
        Exploades dict of Lists into all possible combinations.
        """
        combinations = list(itertools.product(*grid.values()))
        return [dict(zip(grid.keys(), combination)) for combination in combinations]
