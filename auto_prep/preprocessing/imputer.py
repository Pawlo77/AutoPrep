import pandas as pd
from sklearn.impute import SimpleImputer

from ..utils.abstract import Categorical, Numerical
from ..utils.logging_config import setup_logger
from .abstract import NAImputer

logger = setup_logger(__name__)


class NumericalImputer(Numerical, NAImputer):
    """
    Imputer for numerical columns. This class fills missing values in numerical columns
    with the median of the respective column.

    Attributes:
        strategy (str): The imputation strategy (default: "median").
        imputer (SimpleImputer): The SimpleImputer instance for filling missing values.
    """

    def __init__(self):
        super().__init__()
        self.strategy = "median"
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X: pd.DataFrame) -> "NumericalImputer":
        """
        Identifies numeric columns and fits the imputer.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            NumericalImputer: The fitted imputer instance.
        """
        super().fit(X)
        self.numeric_features = [
            col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        ]

        logger.debug(f"Identified numeric features: {self.numeric_features}")
        self.imputer.fit(X[self.numeric_features])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in numeric columns using the median strategy.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with missing numeric values imputed.
        """
        X = super().transform(X)
        X[self.numeric_features] = self.imputer.transform(X[self.numeric_features])
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the input data by imputing missing values in numeric columns.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data with missing numeric values imputed.
        """
        return self.fit(X).transform(X)

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "name": "NumericalImputer",
            "desc": "Imputes numerical missing data.",
            "params": {
                "strategy": self.strategy,
            },
        }


class CategoricalImputer(Categorical, NAImputer):
    """
    Imputer for categorical columns. This class fills missing values in categorical columns
    with the most frequent value in the respective column.

    Attributes:
        strategy (str): The imputation strategy (default: "most_frequent").
        imputer (SimpleImputer): The SimpleImputer instance for filling missing values.
    """

    def __init__(self):
        super().__init__()
        self.strategy = "most_frequent"
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X: pd.DataFrame) -> "CategoricalImputer":
        """
        Identifies categorical columns and fits the imputer.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            CategoricalImputer: The fitted imputer instance.
        """
        super().fit(X)
        self.categorical_features = [
            col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
        ]

        logger.debug(f"Identified categorical features: {self.categorical_features}")
        self.imputer.fit(X[self.categorical_features])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in categorical columns using the most frequent value strategy.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with missing categorical values imputed.
        """
        X = super().transform(X)
        X[self.categorical_features] = self.imputer.transform(
            X[self.categorical_features]
        )
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the input data by imputing missing values in categorical columns.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data with missing categorical values imputed.
        """
        return self.fit(X).transform(X)

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "name": "NumericalImputer",
            "desc": "Imputes categorical missing data.",
            "params": {
                "strategy": self.strategy,
            },
        }
