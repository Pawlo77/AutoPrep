import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..utils.abstract import FeatureImportanceSelector, NonRequiredStep, Numerical
from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class CorrelationSelector(NonRequiredStep, Numerical):
    """
    Transformer to select correlation_percent% (rounded to whole number) of features that are most correlated with the target variable.

    Attributes:
         selected_columns (list): List of selected columns based on correlation with the target.
    """

    def __init__(self):
        """
        Initializes the transformer with a specified percentage of top correlated features to keep.

        Args:
            correlation_percent (float): The percentage of features to retain based on their correlation with the target.
        """
        self.k = config.correlation_percent
        self.selected_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CorrelationSelector":
        """
        Identifies the top correlation_percent% (rounded to whole value) of features most correlated with the target variable.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            CorrelationSelector: The fitted transformer instance.
        """
        logger.start_operation(
            f"Fitting CorrelationSelector with top {self.k}% correlated features."
        )
        try:
            corr_with_target = X.corrwith(y).abs()
            sorted_corr = corr_with_target.sort_values(ascending=False)
            num_top_features = max(1, round(np.ceil(len(sorted_corr) * self.k)))
            self.selected_columns = sorted_corr.head(num_top_features).index.tolist()
        except Exception as e:
            logger.error(f"Error in CorrelationSelector fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top correlation_percent% of features most correlated with the target variable.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% correlated features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most correlated features."
        )
        try:

            X_selected = X[self.selected_columns].copy()
            logger.debug("Successfully transformed CorrelationSelector")
        except Exception as e:
            logger.error(f"Error in CorrelationSelector transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most correlated features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% correlated features."
        )
        try:
            self.fit(X, y)
            X = self.transform(X, y)
        except Exception as e:
            logger.error(f"Error in CorrelationSelector fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "desc": f"Selects the top {self.k*100}% (rounded to whole number) of features most correlated with the target variable. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"correlation_percent": self.k},
        }


class FeatureImportanceClassSelector(FeatureImportanceSelector):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model for classification.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self):
        """
        Initializes the transformer with a specified percentage of top important features to keep.

        """
        super().__init__()
        self.feature_importances_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureImportanceClassSelector":
        """
        Identifies the feature importances according to the Random Forest model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceClassificationSelector: The fitted transformer instance.
        """
        logger.start_operation(
            f"Fitting FeatureImportanceClassificationSelector with top {self.k}% important features."
        )
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            self.feature_importances_ = model.feature_importances_
            total_features = len(self.feature_importances_)
            num_features_to_select = int(np.ceil(total_features * self.k / 100))
            if num_features_to_select == 0:
                num_features_to_select = 1
            indices = np.argsort(self.feature_importances_)[-num_features_to_select:][
                ::-1
            ]
            self.selected_columns = X.columns[indices].tolist()
        except Exception as e:
            logger.error(f"Error in FeatureImportanceClassificationSelector fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most important according to the Random Forest model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most important features."
        )
        try:
            X_selected = X[self.selected_columns].copy()

        except Exception as e:
            logger.error(
                f"Error in FeatureImportanceClassificationSelector transform: {e}"
            )
            raise e
        finally:
            logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.
        """

        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% important features."
        )
        try:
            self.fit(X, y)
            X = self.transform(X, y)
        except Exception as e:
            logger.error(
                f"Error in FeatureImportanceClassificationSelector fit_transform: {e}"
            )
            raise e
        finally:
            logger.end_operation()
        return X

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most important according to Random Forest model for classification. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k},
        }


class FeatureImportanceRegressSelector(FeatureImportanceSelector):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model for regression.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self):
        """
        Initializes the transformer with a specified percentage of top important features to keep.

        """
        super().__init__()
        self.feature_importances_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureImportanceRegressSelector":
        """
        Identifies the feature importances according to the Random Forest model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceRegressionSelector: The fitted transformer instance.
        """
        logger.start_operation(
            f"Fitting FeatureImportanceRegressionSelector with top {self.k}% important features."
        )
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            self.feature_importances_ = model.feature_importances_
            total_features = len(self.feature_importances_)
            num_features_to_select = int(np.ceil(total_features * self.k / 100))
            if num_features_to_select == 0:
                num_features_to_select = 1
            indices = np.argsort(self.feature_importances_)[-num_features_to_select:][
                ::-1
            ]
            self.selected_columns = X.columns[indices].tolist()
        except Exception as e:
            logger.error(f"Error in FeatureImportanceRegressionSelector fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most important according to the Random Forest model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most important features."
        )
        try:

            X_selected = X[self.selected_columns].copy()

        except Exception as e:
            logger.error(f"Error in FeatureImportanceRegressionSelector transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y):
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.
        """

        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% important features."
        )
        try:
            self.fit(X, y)
            X = self.transform(X, y)
        except Exception as e:
            logger.error(
                f"Error in FeatureImportanceRegressionSelector fit_transform: {e}"
            )
            raise e
        finally:
            logger.end_operation()
        return X

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most important according to Random Forest model for regression. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k},
        }
