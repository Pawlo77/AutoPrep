import pandas as pd

from ..utils.abstract import Numerical, RequiredStep
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class NumericalDataImputer(RequiredStep, Numerical):
    """
    Performs Numerical data imputation
    """

    param_grid = {
        "strategy": "median",
    }

    def __init__(self, strategy="median") -> None:
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y=None) -> "NumericalDataImputer":
        """Identify feature types in the dataset.

        Args:
            X (pd.DataFrame): Input features.
            y: Ignored. Exists for scikit-learn compatibility.

        Returns:
            NumericalDataImputer: Fitted transformer.
        Raises:
            ValueError if non numerical column included in X.
        """

        logger.start_operation(f"Numerical data fit ({X.shape[1]} columns).")
        for column in X.columns:
            if not pd.api.types.is_numeric_dtype(X[column]):
                raise ValueError(f"Non numerical feature found: {column}.")
        logger.end_operation()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning and transformation operations to the input data.

        Args:
            X (pd.DataFrame): The input DataFrame to be cleaned and transformed.

        Returns:
            pd.DataFrame: The cleaned and transformed DataFrame.
        """
        try:
            X = X.copy()

            logger.start_operation(f"Numerical data transform (shape - {X.shape}).")
            for col in self.numeric_features:
                missing_count = X[col].isnull().sum()
                if missing_count > 0:
                    logger.debug(
                        f"Imputing {missing_count} missing values in {col} with median."
                    )
                    X[col] = X[col].fillna(X[col].median())
            logger.end_operation()

            return X

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            raise e
