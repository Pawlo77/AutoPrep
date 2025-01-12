import humanize
import numpy as np
import pandas as pd

from ..utils.logging_config import setup_logger
from ..utils.system import get_system_info
from .raport import Raport

logger = setup_logger(__name__)


class OverviewRaport:
    def __init__(self):
        self.dataset_summary: dict = {}
        self.target_distibution: dict = {}
        self.missing_values: dict = {}
        self.features_details: dict = {}
        self.system_info: dict = {}
        self.descr_num: pd.DataFrame = None
        self.descr_cat: pd.DataFrame = None
        self.task: str = None

    def run(self, X: pd.DataFrame, y: pd.Series, task: str):
        """Performs dataset overview."""

        logger.start_operation("Overview.")
        self.task = task

        try:
            self.system_info = get_system_info()
        except Exception as e:
            logger.error(f"Failed to gather system informations: {str(e)}")
            raise e

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

            if task == "classification":
                value_counts = y.value_counts()
                normalized_counts = y.value_counts(normalize=True)
                self.target_distibution = list(
                    zip(
                        value_counts.index,
                        value_counts.values,
                        normalized_counts.values,
                    )
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
            logger.info(
                f"Found {len(numeric_features)} numeric and "
                f"{len(categorical_features)} categorical features"
            )

            if len(numeric_features) > 0:
                self.descr_num = X[numeric_features].describe().T.reset_index()
            if len(categorical_features) > 0:
                self.descr_cat = (
                    X[categorical_features].describe(exclude=["number"]).T.reset_index()
                )

        except Exception as e:
            logger.error(f"Failed to gather overview statistics: {str(e)}")
            raise e

        logger.end_operation()

    def write_to_raport(self, raport: Raport):
        """Writes overview section to a raport"""

        overview_section = raport.add_section("Overview")  # noqa: F841

        system_subsection = raport.add_subsection("System")  # noqa: F841
        raport.add_table(
            self.system_info,
            header=None,
            caption="System overview.",
        )

        dataset_subsection = raport.add_subsection("Dataset")  # noqa: F841
        if self.task == "classification":
            if len(self.target_distibution) > 2:
                raport.add_text(
                    "Task detected for the dataset: multiclass classfication.  \\newline"
                )
            else:
                raport.add_text(
                    "Task detected for the dataset: binary classfication. \\newline"
                )
        elif self.task == "regression":
            raport.add_text("Task detected for the dataset: regression. \\newline")
        raport.add_reference(label="tab:dataset_summary", add_space=True)
        dataset_desc = "presents an overview of the dataset including the number of samples, features, and their types."
        raport.add_text(dataset_desc)

        raport.add_table(
            self.dataset_summary,
            header=None,
            caption="Dataset Summary.",
            label="tab:dataset_summary",
        )

        if len(self.target_distibution) > 0:
            target_desc = "Distribution of the target classes in terms of the number of observations and their percentages is presented in "
            raport.add_text(target_desc)
            raport.add_reference(label="tab:target_distribution", add_space=False)
            raport.add_table(
                self.target_distibution,
                caption="Target class distribution.",
                header=["class", "number of observations", "fraction"],
                label="tab:target_distribution",
            )

        raport.add_reference(label="tab:missing_values", add_space=True)
        missing_values_desc = (
            "presents the distribution of missing values in the dataset."
        )
        raport.add_text(missing_values_desc)

        raport.add_table(
            self.missing_values,
            caption="Missing values distribution.",
            header=["feature", "number of observations", "fraction"],
            label="tab:missing_values",
        )

        raport.add_reference(label="tab:features_dtypes", add_space=True)
        features_desc = "presents the description of features in the dataset."
        raport.add_text(features_desc)
        raport.add_table(
            self.features_details,
            header=["feature", "type", "dtype", "space usage"],
            caption="Features dtypes description.",
            label="tab:features_dtypes",
        )

        columns = [c.replace("%", "\%") for c in self.descr_num.columns]  # noqa W605

        if self.descr_num is not None and self.descr_cat is not None:
            raport.add_reference(label="tab:numerical_features", add_space=True)
            raport.add_text("and ")
            raport.add_reference(label="tab:categorical_features", add_space=True)
            features_desc = "present the description of numerical and categorical features in the dataset."
            raport.add_text(features_desc)
            columns[0] = "feature"
            raport.add_table(
                self.descr_num.values.tolist(),
                header=columns,
                caption="Numerical features description.",
                label="tab:numerical_features",
            )
            raport.add_table(
                self.descr_cat.values.tolist(),
                header=self.descr_cat.columns,
                caption="Categorical features description.",
                label="tab:categorical_features",
            )
        elif self.descr_num is not None:
            raport.add_reference(label="tab:numerical_features", add_space=True)
            features_desc = (
                "presents the description of numerical features in the dataset."
            )
            raport.add_text(features_desc)

            raport.add_table(
                self.descr_num.values.tolist(),
                header=columns,
                caption="Numerical features description.",
                label="tab:numerical_features",
            )
        elif self.descr_cat is not None:
            raport.add_reference(label="tab:categorical_features", add_space=True)
            features_desc = (
                "presents the description of categorical features in the dataset."
            )
            raport.add_text(features_desc)
            self.descr_cat.columns[0] = "feature"
            raport.add_table(
                self.descr_cat.values.tolist(),
                header=self.descr_cat.columns,
                caption="Categorical features description.",
                label="tab:categorical_features",
            )
        else:
            features_desc = "No numerical or categorical features descriptions are available in the dataset."
            raport.add_text(features_desc)

        return raport
