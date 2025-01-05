from abc import ABC, abstractmethod

import pandas as pd

from ..utils.abstract import NonRequiredStep, RequiredStep
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class FeatureImportanceSelector(NonRequiredStep, ABC):
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

    def fit(
        self,
        X,  # noqa: F841
        y,  # noqa: F841
    ):
        """
        Identifies the top k% (rounded to whole value) of features most important according to the model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceSelector: The fitted transformer instance.
        """
        pass

    def transform(
        self,
        X: pd.DataFrame,  # noqa: F841
        y: pd.Series = None,  # noqa: F841
    ):
        """
        Selects the top k% of features most important according to the model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
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

    def is_applicable(
        self,
        dt: pd.DataFrame,  # noqa: F841
    ):
        """
        Args:
            dt (pd.Series) - column that is considered to be preprocessed
                by that transformer.

        Returns:
            bool - True if it is possible to use this transofmation
                on passed data.
        """
        return True


class DimentionReducer(NonRequiredStep, ABC):
    """
    Abstract class for dimensionality reduction techniques.
    """

    def __init__(self):
        super().__init__()
        self.reducer = None

    @abstractmethod
    def fit(self, X, y=None) -> "DimentionReducer":
        pass

    @abstractmethod
    def transform(self, X, y=None) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, X, y=None) -> pd.DataFrame:
        pass

    @abstractmethod
    def is_applicable(X):
        pass

    @abstractmethod
    def to_tex(self) -> dict:
        pass


class NAImputer(RequiredStep, ABC):
    """
    Base class for imputing missing values. Provides functionality
    to identify columns with missing values and determine the strategy to handle them
    (remove columns with >50% missing data).

    Attributes:
        numeric_features (list): A list of numeric feature names.
        categorical_features (list): A list of categorical feature names.
    """

    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []

    def fit(self, X: pd.DataFrame) -> "NAImputer":
        """
        Identifies columns with more than 50% missing values and removes them
        from the dataset.

        Args:
            X (pd.DataFrame): The input data with missing values.

        Returns:
            NAImputer: The fitted imputer instance.
        """
        logger.start_operation(
            f"Fitting NAImputer to data with {X.shape[0]} rows and {X.shape[1]} columns."
        )

        # Removing columns with >50% missing values
        missing_threshold = 0.5
        cols_to_remove = [
            col for col in X.columns if X[col].isnull().mean() > missing_threshold
        ]
        logger.debug(
            f"Columns to be removed due to >50% missing values: {cols_to_remove}"
        )
        # Update internal state but do not modify input DataFrame
        self.cols_to_remove = cols_to_remove

        logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes previously identified columns with >50% missing values.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data.
        """
        return X.drop(columns=self.cols_to_remove, errors="ignore")
