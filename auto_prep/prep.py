import humanize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .raporting.raport import raport
from .utils.config import config
from .utils.logging_config import setup_logger
from .utils.system import get_system_info

logger = setup_logger(__name__)


class AutoPrep:
    """Main pipeline orchestrating the entire preprocessing process.

    This class handles the complete workflow from data preprocessing to
    report generation.
    """

    def __init__(self):
        self.dataset_summary = {}
        self.target_distibution = {}
        self.missing_values = {}
        self.features_details = {}

    def run(self, data: pd.DataFrame, target_column: str):
        """Run the complete pipeline on the provided dataset.

        Args:
            data (pd.DataFrame): Input dataset to process.
            target_column (str): Name of the target variable column.
        """
        logger.info(f"Starting pipeline run with target column: {target_column}")
        logger.debug(f"Input data shape: {data.shape}")

        # "." in names leads to os problems
        data = data.rename(columns=lambda x: x.replace(".", "__"))

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
            logger.start_operation("Overview.")
            self._overview(X_train, y_train)
            logger.end_operation()

            """
            Eda
            """
            logger.start_operation("Eda.")
            self._eda(X_train, y_train)
            logger.end_operation()

            """
            Generate raport
            """
            logger.start_operation("Raport.")
            self._generate_report()
            logger.end_operation()

        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            raise

    def _overview(self, X: pd.DataFrame, y: pd.Series):
        """Performs overview."""

        try:
            numeric_features = X.select_dtypes(include=[np.number]).columns
            categorical_features = X.select_dtypes(exclude=[np.number]).columns

            # Basic statistics
            self.dataset_summary = {
                "Number of samples": len(X),
                "Number of features": len(X.columns),
                "Number of numerical features": len(numeric_features),
                "Number of categorical features": len(categorical_features),
            }

            value_counts = y.value_counts()
            normalized_counts = y.value_counts(normalize=True)
            self.target_distibution = list(
                zip(value_counts.index, value_counts.values, normalized_counts.values)
            )

            missing_value_counts = X.isnull().sum()
            normalized_missing_counts = X.isnull().sum() / len(X)
            self.missing_values = list(
                zip(
                    missing_value_counts.index,
                    missing_value_counts.values,
                    normalized_missing_counts.values,
                )
            )

            self.features_details = [
                (
                    feature,
                    "numerical" if feature in numeric_features else "categorical",
                    X[feature].dtype,
                    humanize.naturalsize(X[feature].memory_usage(deep=True)),
                )
                for feature in X.columns
            ]

            logger.debug(
                f"Found {len(numeric_features)} numeric and "
                f"{len(categorical_features)} categorical features"
            )

        except Exception as e:
            logger.error(f"Failed to generate dataset summary: {str(e)}")
            raise

    def _eda(
        self,
        X: pd.DataFrame,  # noqa: F841
        y: pd.Series,  # noqa: F841
    ):
        """Performs eda."""
        pass  # noqa: F401

    def _generate_report(self):
        """Generates and saves raport."""

        raport.add_header()

        """
        Overview section
        """
        overview_section = raport.add_section("Overview")  # noqa: F841
        system_subsection = raport.add_subsection("System")  # noqa: F841
        raport.add_table(
            get_system_info(),
            header=None,
        )
        dataset_subsection = raport.add_subsection("Dataset")  # noqa: F841
        raport.add_table(
            self.dataset_summary,
            header=None,
        )
        raport.add_table(
            self.target_distibution,
            caption="Target class distribution",
            header=["class", "number of observations", "Percentage"],
        )
        raport.add_table(
            self.missing_values,
            caption="Missing values distribution",
            header=["class", "number of observations", "Percentage"],
        )
        raport.add_table(
            self.features_details,
            header=["class", "type", "dtype", "space usage"],
        )

        raport.generate()
