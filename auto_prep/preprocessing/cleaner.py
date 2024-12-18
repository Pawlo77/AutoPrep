from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Handles missing values, outliers, and basic cleaning operations.

    This transformer identifies feature types and applies appropriate
    cleaning strategies.

    Attributes:
        categorical_threshold (int): Maximum unique values for categorical features.
        numeric_features (List[str]): List of identified numeric feature names.
        categorical_features (List[str]): List of identified categorical feature names.
        high_cardinality_features (List[str]): Features with high unique value counts.
    """

    def __init__(self, categorical_threshold: int = 10) -> None:
        """Initialize the data cleaner.

        Args:
            categorical_threshold (int, optional): Maximum unique values for categorical
                features. Defaults to 10.
        """
        self.categorical_threshold = categorical_threshold
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.high_cardinality_features: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "DataCleaner":
        """Identify feature types in the dataset.

        Args:
            X (pd.DataFrame): Input features.
            y: Ignored. Exists for scikit-learn compatibility.

        Returns:
            DataCleaner: Fitted transformer.
        """
        logger.debug("DataCleaner fit called")
        for column in X.columns:
            n_unique = X[column].nunique()
            if pd.api.types.is_numeric_dtype(X[column]):
                if n_unique <= self.categorical_threshold:
                    self.categorical_features.append(column)
                else:
                    self.numeric_features.append(column)
            else:
                if n_unique > self.categorical_threshold:
                    self.high_cardinality_features.append(column)
                else:
                    self.categorical_features.append(column)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning and transformation operations to the input data.

        Args:
            X (pd.DataFrame): The input DataFrame to be cleaned and transformed.

        Returns:
            pd.DataFrame: The cleaned and transformed DataFrame.
        """
        logger.info("Starting data cleaning transformation")
        logger.debug("Input data shape: %s", X.shape)

        try:
            X = X.copy()

            # Handle missing values in numeric features
            logger.start_operation("numeric missing value imputation")
            for col in self.numeric_features:
                missing_count = X[col].isnull().sum()
                if missing_count > 0:
                    logger.info(
                        f"Imputing {missing_count} missing values in {col} with median"
                    )
                    X[col] = X[col].fillna(X[col].median())
            logger.end_operation()

            # Handle missing values in categorical features
            logger.start_operation("categorical missing value imputation")
            for col in self.categorical_features:
                missing_count = X[col].isnull().sum()
                if missing_count > 0:
                    logger.info(
                        f"Imputing {missing_count} missing values in {col} with mode"
                    )
                    X[col] = X[col].fillna(X[col].mode()[0])
            logger.end_operation()

            # Handle outliers in numeric features
            logger.start_operation("outlier handling")
            for col in self.numeric_features:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = X[col][
                    (X[col] < lower_bound) | (X[col] > upper_bound)
                ].count()
                if outliers > 0:
                    logger.info(f"Clipping {outliers} outliers in {col}")
                    X[col] = X[col].clip(lower_bound, upper_bound)
            logger.end_operation()

            logger.info("Cleaning complete. Output shape: %s", X.shape)
            return X
        except Exception as e:
            logger.error("Data cleaning failed: %s", str(e))
            raise
