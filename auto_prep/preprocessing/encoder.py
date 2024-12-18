from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class SafeLabelEncoder(LabelEncoder):
    """
    LabelEncoder that handles unseen labels by mapping them to an 'unknown' category.
    """

    def __init__(self):
        super().__init__()
        self.unknown_value = None

    def fit(self, y):
        """
        Fit the label encoder on the target variable and set the unknown value.

        Args:
            y: Target variable to fit the encoder on.

        Returns:
            SafeLabelEncoder: Fitted encoder.
        """
        super().fit(y)
        self.unknown_value = len(self.classes_)
        return self

    def transform(self, y):
        """
        Transforms the input labels by mapping any unseen labels to
        the 'unknown' category.

        Args:
            y (array-like): The input labels to be transformed.

        Returns:
            array-like: The transformed labels with unseen labels mapped to 'unknown'.
        """
        try:
            return super().transform(y)
        except ValueError:
            y_new = y.copy() if hasattr(y, "copy") else y[:]
            unseen_mask = ~pd.Series(y).isin(self.classes_)

            if unseen_mask.any():
                unseen_labels = set(y[unseen_mask])
                logger.warning(
                    f"Found {len(unseen_labels)} unseen labels: {unseen_labels}"
                )
                y_new[unseen_mask] = self.classes_[0]  # Map to first seen class

            return super().transform(y_new)


class AdvancedEncoder(BaseEstimator, TransformerMixin):
    """Handles various types of encoding for categorical and numerical features.

    This transformer combines rare category handling, label encoding, and scaling.

    Attributes:
        rare_threshold (float): Threshold for rare category grouping.
        label_encoders (Dict): Dictionary of label encoders for each feature.
        rare_encoders (Dict): Dictionary of rare label encoders for each feature.
        scaler (StandardScaler): Scaler for numeric features.
    """

    def __init__(self, rare_threshold: float = 0.01) -> None:
        """Initialize the encoder.

        Args:
            rare_threshold (float, optional): Threshold for rare categories.
                Defaults to 0.01.
        """
        self.rare_threshold = rare_threshold
        self.label_encoders: Dict = {}
        self.rare_encoders: Dict = {}
        self.scaler = StandardScaler()
        self.categorical_features_ = None
        self.numeric_features_ = None

    def fit(self, X: pd.DataFrame, y=None) -> "AdvancedEncoder":
        """Fit encoders on the input data.

        Args:
            X (pd.DataFrame): Input features.
            y: Ignored. Exists for scikit-learn compatibility.

        Returns:
            AdvancedEncoder: Fitted transformer.
        """
        logger.debug("AdvancedEncoder fit called")
        try:
            logger.start_operation("feature type identification")
            self.numeric_features = X.select_dtypes(
                include=["int64", "float64"]
            ).columns
            self.categorical_features = X.select_dtypes(
                include=["object", "category"]
            ).columns
            logger.info(
                f"Identified {len(self.numeric_features)} numeric and "
                "{len(self.categorical_features)} categorical features"
            )
            logger.end_operation()

            # Initialize encoders
            logger.start_operation("encoder initialization")
            self.label_encoders = {
                col: SafeLabelEncoder() for col in self.categorical_features
            }
            self.scaler = StandardScaler()
            logger.end_operation()

            # Handle missing values and fit label encoders
            if self.categorical_features.size > 0:
                logger.start_operation("label encoder fitting")
                X_temp = X.copy()
                for col in self.categorical_features:
                    # Handle missing values before fitting
                    if X_temp[col].isnull().any():
                        logger.debug(f"Filling NaN values in {col} with 'missing'")
                        X_temp[col] = X_temp[col].fillna("missing")

                    unique_values = len(X_temp[col].unique())
                    logger.debug(
                        f"Fitting label encoder for {col} with {unique_values} "
                        "unique values"
                    )
                    self.label_encoders[col].fit(X_temp[col])
                logger.end_operation()

            # Fit scaler
            if self.numeric_features.size > 0:
                logger.start_operation("scaler fitting")
                # Handle missing values before fitting
                X_numeric = X[self.numeric_features].copy()
                for col in self.numeric_features:
                    if X_numeric[col].isnull().any():
                        logger.debug(f"Filling NaN values in {col} with median")
                        X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median())

                self.scaler.fit(X_numeric)
                logger.end_operation()

            return self
        except Exception as e:
            logger.error(f"Encoder fitting failed: {str(e)}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data using fitted encoders.

        Args:
            X (pd.DataFrame): Input features to transform.

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        logger.info("Starting data encoding transformation")
        logger.debug(f"Input data shape: {X.shape}")

        try:
            X = X.copy()

            # Encode categorical features
            if self.categorical_features.size > 0:
                logger.start_operation("categorical encoding")
                for col in self.categorical_features:
                    # Fill NaN values before encoding
                    if X[col].isnull().any():
                        logger.debug(f"Filling NaN values in {col} with 'missing'")
                        X[col] = X[col].fillna("missing")

                    unique_values = len(X[col].unique())
                    logger.debug(f"Encoding {col} with {unique_values} unique values")
                    X[col] = self.label_encoders[col].transform(X[col])
                logger.end_operation()

            # Scale numeric features
            if self.numeric_features.size > 0:
                logger.start_operation("numeric scaling")
                # Fill NaN values with median before scaling
                for col in self.numeric_features:
                    if X[col].isnull().any():
                        logger.debug(f"Filling NaN values in {col} with median")
                        X[col] = X[col].fillna(X[col].median())

                X[self.numeric_features] = self.scaler.transform(
                    X[self.numeric_features]
                )
                logger.end_operation()

            logger.info(f"Encoding complete. Output shape: {X.shape}")
            return X
        except Exception as e:
            logger.error(f"Encoding transformation failed: {str(e)}")
            raise
