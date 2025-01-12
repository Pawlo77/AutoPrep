from abc import ABC, abstractmethod

import pandas as pd

from ..utils.abstract import NonRequiredStep, Numerical
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class DimentionReducer(NonRequiredStep, Numerical, ABC):
    """
    Abstract class for dimensionality reduction techniques.
    """

    def __init__(self):
        super().__init__()
        self.reducer = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "DimentionReducer":
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def to_tex(self) -> dict:
        pass


class FeatureImportanceSelector(NonRequiredStep):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self, k: float = 10.0):
        """
        Initializes the transformer with a specified model and percentage of top important features to keep.

        Args:
            k (float): The percentage of features to retain based on their importance.
        """
        if not (0 <= k <= 100):
            raise ValueError("k must be between 0 and 100.")
        self.k = k
        self.selected_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureImportanceSelector":
        """
        Identifies the top k% (rounded to whole value) of features most important according to the model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceSelector: The fitted transformer instance.
        """
        pass

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most important according to the model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        self.fit(X, y)
        return self.transform(X)
