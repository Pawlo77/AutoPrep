import json
import logging
import os
from time import time
from typing import List, Union

import humanize
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from ..utils.abstract import Classifier, ModulesHandler, Regressor
from ..utils.config import config
from ..utils.logging_config import setup_logger
from ..utils.other import save_model

logger = setup_logger(__name__)

format_shape: callable = lambda df: f"{df.shape[0]} samples, {df.shape[1]} features"


def custom_sort(key_value):
    key, _ = key_value
    key_lower = key.lower()

    # Check if key ends with "time"
    if key_lower.endswith("time"):
        return (2, key)  # Time keys come last
    # Check if key contains "score"
    elif "score" in key_lower:
        return (1, key)  # Score keys come after regular keys
    else:
        return (0, key)


class ModelHandler:
    """
    Class responsible for loading and handling machine learning models and pipelines.
    """

    def __init__(self):
        self._task: str = None
        self._data_meta: dict = {}
        self._model_meta: List[dict] = []
        self._unique_models_params_checked: int = 0
        self._scoring_func = None

        self._models_classes: List[BaseEstimator] = []
        self._pipelines: List[BaseEstimator] = []
        self._pipelines_names: List[str] = []
        self._results: List[dict] = []
        self._stats: List[dict] = []
        self._best_models_results: List[dict] = []

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task: str,
    ):
        """
        Performs models fitting and selection.

        Args:
            X_train (pd.DataFrame): Training feature dataset.
            y_train (pd.Series): Training target dataset.
            X_valid (pd.DataFrame): Validation feature dataset.
            y_valid (pd.Series): Validation target dataset.
            X_test (pd.DataFrame): Test feature dataset.
            y_test (pd.Series): Test target dataset.
            task (str): regiression / classification
        """
        self._task = task
        self._data_meta = {
            "train": format_shape(X_train),
            "valid": format_shape(X_valid),
            "test": format_shape(X_test),
        }

        self._models_classes = ModelHandler.load_models(task)
        pipelines, pipelines_file_names = ModelHandler.load_pipelines()

        self._scoring_func = (
            config.classification_pipeline_scoring_func
            if task == "classification"
            else config.regression_pipeline_scoring_func
        )

        logger.start_operation("Tuning models...")
        logger.info(
            f"Will train {len(self._models_classes)} for each of {len(pipelines)} preprocessing pipelines."
        )
        for idx, (pipeline, pipeline_file_name) in enumerate(
            zip(pipelines, pipelines_file_names)
        ):
            try:
                X_train_cur = pipeline.transform(X_train)
                X_valid_cur = pipeline.transform(X_valid)
            except Exception as e:
                raise Exception(
                    f"Faulty preprocessing pipeline {pipeline_file_name}"
                ) from e

            gen = self._models_classes
            if logger.level >= logging.INFO:
                gen = tqdm(
                    gen, desc=f"Tuning models for pipeline number {idx}", unit="model"
                )
            for model_cls in gen:
                try:
                    info, results, n_runs = ModelHandler.tune_model(
                        scoring_func=self._scoring_func,
                        model_cls=model_cls,
                        best_k=config.max_models,
                        X_train=X_train_cur,
                        y_train=y_train,
                        X_valid=X_valid_cur,
                        y_valid=y_valid,
                    )
                except Exception as e:
                    raise Exception(f"Failed to tune {model_cls.__name__}") from e

                info["Preprocessing pipeline name"] = pipeline_file_name
                for r in results:
                    r["Preprocessing pipeline name"] = pipeline_file_name
                    r["Preprocessing pipeline"] = pipeline
                    r["Model cls"] = model_cls

                self._stats.append(info)
                self._results.extend(results)

                if idx == 0:
                    self._model_meta.append(
                        {
                            "name": model_cls.__name__,
                            "unique params distributions checked": n_runs,
                        }
                    )
                    self._unique_models_params_checked += n_runs

        logger.end_operation()

        self._results = sorted(
            self._results,
            key=lambda x: (
                x["mean_test_score"],
                x["std_test_score"],
                -x["std_fit_time"],
            ),
        )

        logger.start_operation("Re-training best models...")
        logger.info(f"Re-training for up to {config.max_models} best models.")
        gen = self._results[: config.max_models]
        if logger.level >= logging.INFO:
            gen = tqdm(gen, desc="Re-training best models...", unit="model")
        for idx, result in enumerate(gen):
            model_cls = result.pop("Model cls")
            pipeline = result.pop("Preprocessing pipeline")
            pipeline_file_name = result.pop("Preprocessing pipeline name")

            X_train_cur = pipeline.transform(X_train)
            X_valid_cur = pipeline.transform(X_valid)
            X_test_cur = pipeline.transform(X_test)

            model = model_cls(**result["params"])

            X_combined = np.vstack([X_train_cur, X_valid_cur])
            y_combined = pd.concat([y_train, y_test], axis=0)

            t0 = time()
            model.fit(X_combined, y_combined)
            result["re-training time"] = time() - t0

            y_combined_pred = model.predict(X_combined)
            y_test_pred = model.predict(X_test_cur)

            combined_score = self._scoring_func(y_combined, y_combined_pred)
            test_score = self._scoring_func(y_test, y_test_pred)

            result["name"] = model_cls.__name__
            result["params"] = json.dumps(result["params"])
            result["combined score (after re-training)"] = combined_score
            result["test score (after re-training)"] = test_score

            final_pipeline_name = f"final_pipeline_{idx}.joblib"
            result["final pipeline name"] = final_pipeline_name
            self._best_models_results.append(result)

            final_model = Pipeline([("preprocessing", pipeline), ("predicting", model)])
            save_model(final_pipeline_name, final_model)

        logger.end_operation()

    def write_to_raport(self, raport):
        """Writes overview section to a raport"""

        modeling_section = raport.add_section("Modeling")  # noqa: F841

        section_desc = f"This part of the report presents the results of the modeling process. It was configured to create up to {config.max_models} models."
        raport.add_text(section_desc)

        raport.add_subsection("Overview")
        overview = {
            "task": self._task,
            "unique models param sets checked (for each dataset)": self._unique_models_params_checked,
            "unique models": len(self._models_classes),
            "scoring function": self._scoring_func.__name__,
            "search parameters": json.dumps(config.tuning_params),
            **self._data_meta,
        }
        raport.add_table(
            overview, caption="General input data overview.", widths=[40, 120]
        )

        model_meta = pd.DataFrame(self._model_meta)
        raport.add_table(
            model_meta.values.tolist(),
            caption="Used models.",
            header=model_meta.columns,
        )

        for idx, model_results in enumerate(self._best_models_results):
            raport.add_subsection(f"Scores for {idx}th best model")

            for k in model_results.keys():
                if k.endswith("time"):
                    model_results[k] = humanize.naturaldelta(model_results[k])
                elif "score" in k.lower():
                    model_results[k] = round(
                        model_results[k], config.raport_decimal_precision
                    )

            model_results = [
                f"{k}: {v}" for k, v in sorted(model_results.items(), key=custom_sort)
            ]
            raport.add_list(model_results)

        return raport

    @staticmethod
    def load_models(task: str) -> List[BaseEstimator]:
        logger.start_operation("Loading models...")
        package = ModulesHandler.get_subpackage(__file__)
        modules = ModelHandler.load_modules(package=os.path.dirname(__file__))

        classes = []
        for module in modules:
            classes.extend(
                ModulesHandler.load_classes(module_name=module, package=package)
            )

        models_classes = []
        for classes in classes:
            if task == "regression" and issubclass(classes, Regressor):
                models_classes.append(classes)
            elif task == "classification" and issubclass(classes, Classifier):
                models_classes.append(classes)

        logger.debug(f"Loaded {models_classes} models.")
        logger.end_operation()
        return models_classes

    @staticmethod
    def load_modules(package: str) -> List[str]:
        """
        Loads modules from the specified package that contains models
        (start with model_).

        Args:
            package (str): The package to load modules from.
        Returns:
            List[str]: found module names.
        """
        modules = []
        for file_name in os.listdir(package):
            if file_name.startswith("model_") and file_name.endswith(".py"):
                modules.append(f".{os.path.splitext(file_name)[0]}")
        logger.debug(f"Found model modules: {modules}")
        return modules

    def load_pipelines() -> Union[List[BaseEstimator], List[str]]:
        """
        Loads pipelines from the directory specified in config.

        Returns:
            List[BaseEstimator]: loaded pipelines.
            List[str]: pipelines file names.
        """
        logger.start_operation("Loading pipelines...")

        pipelines = []
        file_names = []

        for file_name in os.listdir(config.pipelines_dir):
            if file_name.endswith(".joblib") and file_name.startswith("preprocessing_"):
                file_names.append(file_name)

        file_names = sorted(file_names)
        try:
            for file_name in file_names:
                pipeline = joblib.load(os.path.join(config.pipelines_dir, file_name))
                pipelines.append(pipeline)
            return pipelines, file_names
        except Exception as e:
            logger.error(f"Error in loading pipelines: {e}")
            raise e
        finally:
            logger.end_operation()

    @staticmethod
    def tune_model(
        scoring_func: callable,
        model_cls: BaseEstimator,
        best_k: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ) -> Union[dict, List[dict], int]:
        """
        Tunes a model's hyperparameters using RandomizedSearchCV and returns the best model and related information.

        Args:
            scoring_func (Callable): Scoring function for evaluating models.
            model_cls (BaseEstimator): Model class to be trained.
            best_k (int): Return up to k best models params.
            X_train (pd.DataFrame): Training feature dataset.
            y_train (pd.Series): Training target dataset.
            X_valid (pd.DataFrame, optional): Validation feature dataset. Defaults to None.
            y_valid (pd.Series, optional): Validation target dataset. Defaults to None.

        Returns:
            dict: training meta info
            List[dict]: results
            int: models tested
        """
        if not hasattr(model_cls, "PARAM_GRID"):
            raise AttributeError("Model class must define a PARAM_GRID attribute.")

        logger.debug(f"Tuning model {model_cls.__name__}")

        random_search = RandomizedSearchCV(
            estimator=model_cls(),
            param_distributions=model_cls.PARAM_GRID,
            scoring=scoring_func,
            **config.tuning_params,
        )

        t0 = time()

        # Fit with or without validation set
        fit_params = {}
        if X_valid is not None and y_valid is not None:
            if hasattr(model_cls, "eval_set"):
                fit_params.update(
                    {
                        "eval_set": [(X_valid, y_valid)],
                        "eval_metric": scoring_func,
                        "early_stopping_rounds": 10,
                    }
                )
        random_search.fit(X_train, y_train, **fit_params)

        info = {
            "search_time": time() - t0,
            "best_score": random_search.best_score_,
            "best_index": random_search.best_index_,
        }
        results = pd.DataFrame(random_search.cv_results_)
        sorted_results = results.sort_values(by="mean_test_score", ascending=False)
        top_models_stats = sorted_results.head(best_k)[
            [
                "params",
                "mean_test_score",
                "std_test_score",
                "mean_fit_time",
                "std_fit_time",
            ]
        ].to_dict(orient="records")

        return info, top_models_stats, len(results)
