import pandas as pd
from sklearn.impute import SimpleImputer

from ..utils.abstract import NumericalCategorical, RequiredStep
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class NAImputer(RequiredStep, NumericalCategorical):
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
        self.cols_to_remove = []
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.numerical_imputer = SimpleImputer(strategy="median")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "NAImputer":
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

        try:
            missing_threshold = 0.5
            cols_to_remove = [
                col for col in X.columns if X[col].isnull().mean() > missing_threshold
            ]
            logger.debug(
                f"Columns to be removed due to >50% missing values: {cols_to_remove}"
            )
            # Update internal state but do not modify input DataFrame
            self.cols_to_remove = cols_to_remove
            self.numeric_features = X.select_dtypes(include="number").columns.tolist()
            logger.debug(f"Identified numeric features: {self.numeric_features}")
            self.categorical_features = X.select_dtypes(
                exclude="number"
            ).columns.tolist()
            logger.debug(
                f"Identified categorical features: {self.categorical_features}"
            )
        except Exception as e:
            logger.error(f"Error in NumericalImputer fit: {e}")
            raise e
        finally:
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

        logger.start_operation("Transforming data.")
        try:
            X = X.drop(columns=self.cols_to_remove)

            # Impute missing values in numeric columns
            available_numeric_features = [
                col for col in self.numeric_features if col in X.columns
            ]
            if available_numeric_features:
                self.numerical_imputer.fit(X[available_numeric_features])
                X[available_numeric_features] = self.numerical_imputer.transform(
                    X[available_numeric_features]
                )

            # Impute missing values in categorical columns
            available_categorical_features = [
                col for col in self.categorical_features if col in X.columns
            ]
            if available_categorical_features:
                self.categorical_imputer.fit(X[available_categorical_features])
                X[available_categorical_features] = self.categorical_imputer.transform(
                    X[available_categorical_features]
                )

            for col in available_categorical_features:
                X[col] = X[col].fillna("Missing")
            for col in available_numeric_features:
                X[col] = X[col].fillna(X[col].median())
        except Exception as e:
            logger.error(f"Error in NAImputer transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits and transforms the input data by imputing missing values.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data with missing values imputed.
        """
        logger.start_operation("Fitting and transforming data.")
        try:
            self.fit(X)
            X = self.transform(X)
        except Exception as e:
            logger.error(f"Error in NAImputer fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "desc": "Imputes missing data.",
            "params": {
                "numeric_imputer": self.numerical_imputer.strategy,
                "categorical_imputer": self.categorical_imputer.strategy,
            },
        }
