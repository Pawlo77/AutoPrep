from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from .modeling.handler import ModelHandler
from .preprocessing.handler import PreprocessingHandler
from .raporting.eda import EdaRaport
from .raporting.overview import OverviewRaport
from .raporting.raport import Raport
from .utils.config import config
from .utils.logging_config import setup_logger

logger = setup_logger(__name__)


format_column_name: Callable = lambda x: x.replace(".", "__")


class AutoPrep:
    """Main pipeline orchestrating the entire preprocessing process.

    This class handles the complete workflow from data preprocessing to
    report generation.
    """

    def __init__(self):
        self.overview_raport: OverviewRaport = OverviewRaport()
        self.eda_raport: EdaRaport = EdaRaport()
        self.preprocessing_handler: PreprocessingHandler = PreprocessingHandler()
        self.model_handler: ModelHandler = ModelHandler()

    def run(self, data: pd.DataFrame, target_column: str):
        """Run the complete pipeline on the provided dataset.

        Args:
            data (pd.DataFrame): Input dataset to process.
            target_column (str): Name of the target variable column.
        """
        logger.info(f"Starting pipeline run with target column: {target_column}")
        logger.debug(f"Input data shape: {data.shape}")

        # "." in names leads to os problems
        for col in data.columns:
            if "." in col:
                logger.warning(
                    f"Column '{col}' will be renamed to {format_column_name(col)}"
                    "renamed due to '.' in it's name, which leads to os problems."
                )
        data = data.rename(columns=lambda x: x.replace(".", "__"))

        self._run(data, target_column)

        self._generate_report()

    def _run(self, data: pd.DataFrame, target_column: str, task: str = "auto"):
        """
        Performs all neccessary computations.

        Args:
            data (pd.DataFrame): Input dataset to process.
            target_column (str): Name of the target variable column.
            task (str): Classification or regression. For auto will try to detect it itself.
        """

        logger.start_operation("Calculations.")

        if task == "auto":
            task = AutoPrep._detect_task_type(data, target_column)
            logger.info(f"Detected task: {task}")
        else:
            assert task in ["regression", "classification"], "Unknown task."

        try:
            """
            Split data
            """
            logger.start_operation("Spliting data.")

            # Split features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, train_size=config.train_size, random_state=config.random_state
            )
            X_valid, X_test, y_valid, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=config.test_size / (config.test_size + config.valid_size),
                random_state=config.random_state,
            )  # noqa: F841
            logger.end_operation()

            """
            Overview
            """
            self.overview_raport.run(X_train, y_train)

            """
            Eda
            """
            self.eda_raport.run(X_train, y_train)

            """
            Preprocessing
            """
            self.preprocessing_handler.run(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                task=task,
            )

            """
            Modeling
            """
            self.model_handler.run(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                task=task,
            )

        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            raise

        logger.end_operation()

    def _generate_report(self):
        """Generates and saves raport."""

        raport = Raport()

        logger.start_operation("Generate raport.")
        raport.add_header()

        """
        Overview
        """
        self.overview_raport.write_to_raport(raport)

        """
        Eda
        """
        self.eda_raport.write_to_raport(raport)

        """
        Preprocessing
        """
        self.preprocessing_handler.write_to_raport(raport)

        """
        Modeling
        """
        self.model_handler.write_to_raport(raport)

        raport.generate()
        logger.end_operation()

    @staticmethod
    def _detect_task_type(df: pd.DataFrame, target_column: str):
        """
        Automatically detects whether the target column is for regression or classification.

        Args:
            df (pd.DataFrame): The input dataset.
            target_column (str): The name of the target column.

        Returns:
            str: 'regression' or 'classification' based on the target column.
        """
        target = df[target_column]

        if pd.api.types.is_numeric_dtype(target):
            unique_values = target.nunique()

            if unique_values < config.max_unique_values_classification:
                return "classification"
            else:
                return "regression"
        return "classification"
