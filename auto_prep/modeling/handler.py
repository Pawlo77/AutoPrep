import glob
import json
import logging
import os
import random
from time import time
from typing import List, Union

import humanize
import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from ..utils.abstract import Classifier, ModulesHandler, Regressor
from ..utils.config import config
from ..utils.logging_config import setup_logger
from ..utils.other import save_chart, save_model

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

        if task == "classification":
            if len(y_train.unique()) > 2:
                self._scoring_func = config.classification_pipeline_scoring_func_multi
            else:
                self._scoring_func = config.classification_pipeline_scoring_func
        else:
            self._scoring_func = config.regression_pipeline_scoring_func

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
                        task=task,
                    )
                except Exception as e:
                    raise Exception(f"Failed to tune {model_cls.__name__}") from e

                info["Preprocessing pipeline name"] = pipeline_file_name
                for r in results:
                    r["Preprocessing pipeline name"] = pipeline_file_name
                    r["Preprocessing pipeline"] = pipeline
                    r["Model cls"] = model_cls
                    r["Model cls base name"] = model_cls.__bases__[0].__name__

                self._stats.append(info)
                self._results.extend(results)

                if idx == 0:
                    self._model_meta.append(
                        {
                            "name": model_cls.__name__,
                            "unique params distributions checked": n_runs,
                            "param_grid": model_cls.PARAM_GRID,
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
            y_combined = pd.concat([y_train, y_valid], axis=0)

            if pd.api.types.is_categorical_dtype(y_combined):
                label_encoder = LabelEncoder()
                label_encoder.fit(y_combined)
                y_combined = label_encoder.transform(y_combined)

            t0 = time()
            model.fit(X_combined, y_combined)
            result["re-training time"] = time() - t0
            y_combined_pred = model.predict(X_combined)
            y_test_pred = model.predict(X_test_cur)

            combined_score = self._scoring_func(y_combined, y_combined_pred)
            test_score = self._scoring_func(y_test, y_test_pred)

            result["model"] = model
            result["name"] = model_cls.__name__
            result["params"] = json.dumps(result["params"])
            result["combined score (after re-training)"] = combined_score
            result["test score (after re-training)"] = test_score
            result["model base name"] = model_cls.__bases__[0].__name__

            final_pipeline_name = f"final_pipeline_{idx}.joblib"
            result["final pipeline name"] = final_pipeline_name
            self._best_models_results.append(result)

            final_model = Pipeline([("preprocessing", pipeline), ("predicting", model)])
            save_model(final_pipeline_name, final_model)

        best_model_result = max(
            self._best_models_results, key=lambda x: x["test score (after re-training)"]
        )
        self.genarate_shap(X_test_cur, best_model_result["model"], 0)

        logger.end_operation()

    def write_to_raport(self, raport):
        """Writes overview section to a raport"""

        modeling_section = raport.add_section("Modeling")  # noqa: F841

        raport.add_subsection("Overview")
        overview = {
            "task": self._task,
            "unique models param sets checked (for each dataset)": self._unique_models_params_checked,
            "unique models": len(self._models_classes),
            "base model names": [
                model_cls.__bases__[0].__name__ for model_cls in self._models_classes
            ],
            "scoring function": self._scoring_func.__name__,
            "search parameters": json.dumps(config.tuning_params),
            "results": self._results,
            "best models results": self._best_models_results,
            **self._data_meta,
        }
        section_desc = f"This part of the report presents the results of the modeling process. There were {overview['unique models']} {overview['task']} models trained for each of the best preprocessing pipelines. \\newline"
        raport.add_text(section_desc)
        used_models_desc = (
            "The following models were used in the modeling process. \\newline"
        )
        raport.add_text(used_models_desc)

        raport.add_list(overview["base model names"])

        raport.add_subsection("Hyperparameter tuning")
        section_desc = f"This section presents the results of hyperparameter tuning for each of the best {config.max_models} models using RandomizedSearchCV. Param grids used for each model are presented in the tables below. "
        raport.add_text(section_desc)

        model_meta = pd.DataFrame(self._model_meta)

        for model, param_grid in model_meta[["name", "param_grid"]].values.tolist():
            model_base_name = model[5:]
            if isinstance(param_grid, dict):
                raport.add_table(
                    param_grid,
                    caption=f"Param grid for model {model_base_name}.",
                )
            else:
                raport.add_table(
                    {i: json.dumps(v) for i, v in enumerate(param_grid)},
                    caption=f"Param grid for model {model_base_name}.",
                )

        results_df = pd.DataFrame(overview["results"])
        results_df.to_csv("results.csv", index=False)
        best_results = pd.DataFrame(overview["best models results"])

        best_results = (
            best_results[
                [
                    "Model cls base name",
                    "final pipeline name",
                    "params",
                    "mean_fit_time",
                    "test score (after re-training)",
                ]
            ]
            .rename(
                columns={
                    "Model cls base name": "Model",
                    "final pipeline name": "Pipeline",
                    "params": "Best params",
                    "mean_fit_time": "Mean fit time",
                    "test score (after re-training)": "Test score",
                }
            )
            .sort_values(by="Test score", ascending=overview["task"] == "regression")
        )
        best_results["Mean fit time"] = best_results["Mean fit time"].apply(
            lambda x: humanize.naturaldelta(x)
        )
        best_results_dict = list(best_results.itertuples(index=False, name=None))

        raport.add_reference(label="tab:best_models_results", add_space=True)
        best_models_desc = "presents the best models and pipelines along with their hyperparameters, mean fit time, and test score."
        raport.add_text(best_models_desc)
        raport.add_table(
            best_results_dict,
            caption="Best models results",
            header=["Model", "Pipeline", "Best params", "Mean fit time", "Test score"],
            widths=[35, 35, 45, 25, 15],
            label="tab:best_models_results",
        )

        raport.add_subsection("Interpretability")
        interpretability_desc = "This section presents SHAP plots for the best model."
        charts_dir = config.charts_dir
        raport.add_text(interpretability_desc)

        shap_order = ["shap_bar", "shap_summ", "shap_wat"]
        logger.start_operation("Adding SHAP plots to raport...")
        try:
            for order in shap_order:
                for file in glob.glob(os.path.join(charts_dir, "shap*.png")):
                    if order in file:
                        if overview["task"] == "classification":
                            class_no = file.split("_")[2]
                            if order == "shap_summ":
                                caption = f"SHAP summary plot for class {class_no}."
                            elif order == "shap_wat":
                                caption = f"SHAP waterfall plot for class {class_no}."
                            else:
                                caption = f"SHAP bar plot for class {class_no}."
                        else:
                            if order == "shap_summ":
                                caption = "SHAP summary plot."
                            elif order == "shap_wat":
                                caption = "SHAP waterfall plot."
                            else:
                                caption = "SHAP bar plot."
                        raport.add_figure(file, caption=caption)
        except Exception as e:
            logger.error(f"Error in adding SHAP plots to raport: {e}")
        finally:
            logger.end_operation()

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
        task: str = None,
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
        results.to_csv("results.csv", index=False)
        sorted_results = results.sort_values(by="mean_test_score", ascending=True)
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

    @staticmethod
    def genarate_shap(X_test: pd.DataFrame, model: BaseEstimator, model_idx: int):
        """
        Generates SHAP plots for the best model.

        Args:
            X_test (pd.DataFrame): The input DataFrame containing the test data.
            model (BaseEstimator): The trained model used for prediction.
            model_idx (int): The index of the model.

        """

        def clean_shap_files():
            charts_dir = config.charts_dir
            for file in glob.glob(os.path.join(charts_dir, "shap*.png")):
                os.remove(file)
                logger.info(f"Removed SHAP files from {charts_dir}.")

        clean_shap_files()
        logger.info(f"SHAP - X test columns : {X_test.columns}")
        background = shap.sample(X_test, min(100, len(X_test)))
        n = len(X_test)
        sample_size = int(0.5 * n)
        logger.info(f"X_test sample for shap created, sample size: {sample_size}")
        X_sample = X_test.sample(n=sample_size, random_state=42)
        feature_names = X_sample.columns.to_list()
        background_np = (
            background.values.astype(np.float64)
            if isinstance(background, pd.DataFrame)
            else background
        )
        X_sample_np = (
            X_sample.values.astype(np.float64)
            if isinstance(X_sample, pd.DataFrame)
            else X_sample
        )

        try:
            if hasattr(model, "predict_proba"):
                explainer = shap.Explainer(model.predict_proba, background_np)
                logger.info("SHAP explainer created")
                shap_values = explainer(X_sample)
                logger.info("SHAP values created")

            elif hasattr(model, "predict"):
                explainer = shap.Explainer(model.predict, background_np)
                logger.info("SHAP explainer created")
                shap_values = explainer(X_sample_np)
                shap_values.feature_names = feature_names
                logger.info("SHAP values created")

            else:
                raise TypeError(
                    "Model must have either 'predict_proba' or 'predict' method to use SHAP."
                )

            sample_idx = random.randint(0, X_sample.shape[0] - 1)

            if len(shap_values.values.shape) == 3:
                logger.info("SHAP - classification")
                num_classes = shap_values.values.shape[2]
                logger.info(f" SHAP - Number of classes:{num_classes}")

                for class_idx in range(num_classes):
                    logger.info(
                        f"\nGenerating plots for model : {model_idx}, for Class {class_idx}:"
                    )

                    shap.summary_plot(shap_values[..., class_idx], X_sample, show=False)
                    plt.title(f"Summary plot for class {class_idx}")
                    plt.tight_layout()
                    shap_sum = save_chart(
                        f"shap_summ_{class_idx}_cls.png"
                    )  # noqa: F841

                    shap.waterfall_plot(
                        shap_values[:, :, class_idx][sample_idx], show=False
                    )
                    plt.title(
                        f"Waterfall plot for class {class_idx}, observation numer: {sample_idx}"
                    )
                    plt.tight_layout()
                    shap_waterfall = save_chart(
                        f"shap_wat_{class_idx}_cls.png"
                    )  # noqa: F841

                    shap.plots.bar(
                        shap_values[..., class_idx], max_display=10, show=False
                    )
                    plt.title(f"Bar plot for class {class_idx}")
                    plt.tight_layout()
                    shap_bar = save_chart(f"shap_bar_{class_idx}_cls.png")  # noqa: F841

                logger.info(f"SHAP plots saved for model : {model_idx}")
            else:
                logger.info(f"\nGenerating plots for model: {model_idx}... Regression")

                shap.summary_plot(
                    shap_values, X_sample, feature_names=feature_names, show=False
                )
                plt.title("Summary plot for regression")
                plt.tight_layout()
                reg_sum = save_chart(f"shap_summ_regression.png")  # noqa: F841

                shap_values.feature_names = [str(name) for name in shap_values.feature_names]
                shap.waterfall_plot(shap_values[sample_idx], show=False)
                plt.title(
                    f"Waterfall plot for regression, observation numer: {sample_idx}"
                )
                plt.tight_layout()
                reg_waterfall = save_chart(f"shap_wat_regression.png")  # noqa: F841

                shap.plots.bar(shap_values, max_display=10, show=False)
                plt.title("Bar plot for regression")
                plt.tight_layout()
                reg_bar = save_chart(f"shap_bar_regression.png")  # noqa: F841

                logger.info(f"SHAP plots saved for model : {model_idx}")

        except Exception as e:
            logger.error(f"Shap failed : {e}")
            raise
