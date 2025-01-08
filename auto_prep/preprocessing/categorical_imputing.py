import pandas as pd
from sklearn.impute import SimpleImputer

from ..utils.abstract import Categorical
from ..utils.config import config
from ..utils.logging_config import setup_logger
from .abstract import NAImputer

logger = setup_logger(__name__)

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
        self.strategy = config.imputer_settings["categorical_strategy"]
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.categorical_features = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """
        Identifies categorical columns and fits the imputer.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            CategoricalImputer: The fitted imputer instance.
        """
        logger.start_operation(
            f"Fitting CategoricalImputer to data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        try:
            self.categorical_features = X.select_dtypes(
                exclude="number"
            ).columns.tolist()
            logger.debug(
                f"Identified categorical features: {self.categorical_features}"
            )
        except Exception as e:
            logger.error(f"Error in CategoricalImputer fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in categorical columns using the most frequent value strategy.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with missing categorical values imputed.
        """
        logger.start_operation("Transforming data.")
        try:
            super().fit(X)
            X = super().transform(X)
            available_features = [
                col for col in self.categorical_features if col in X.columns
            ]
            self.imputer.fit(X[available_features])
            X[available_features] = self.imputer.transform(X[available_features])

            for col in available_features:
                X[col].fillna("Missing", inplace=True)
                
            # print how many nans per column
            logger.debug(X.isnull().sum())
            
        except Exception as e:
            logger.error(f"Error in CategoricalImputer transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits and transforms the input data by imputing missing values in categorical columns.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data with missing categorical values imputed.
        """

        logger.start_operation("Fitting and transforming data.")
        try:
            self.fit(X)
            X = self.transform(X)
        except Exception as e:
            logger.error(f"Error in CategoricalImputer fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "desc": "Imputes categorical missing data.",
            "params": {
                "strategy": self.strategy,
            },
        }
