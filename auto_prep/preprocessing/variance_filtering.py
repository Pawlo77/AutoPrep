import pandas as pd

from ..utils.abstract import Numerical, RequiredStep
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class VarianceFilter(RequiredStep, Numerical):
    """
    Transformer to remove numerical columns with zero variance.

    Attributes:
        dropped_columns (list): List of dropped columns.
    """

    def __init__(self):
        """
        Initializes the transformer with empty list of dropped columns.
        """

        self.dropped_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "VarianceFilter":
        """
        Identifies columns with zero variances and adds to dropped_columns list.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            VarianceAndUniqueFilter: The fitted transformer instance.
        """
        logger.start_operation("Fitting VarianceFilter")
        try:
            zero_variance = X.var() == 0
            self.dropped_columns = X.columns[zero_variance].tolist()
            logger.debug(
                f"Successfully fitted VarianceFilter, columns with 0 variance: {self.dropped_columns}"
            )
        except Exception as e:
            logger.error(f"Failed to fit VarianceFilter : {e}", exc_info=True)
            raise ValueError(f"Failed to fit VarianceFilter: {e}")
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Drops the identified columns with zero variance based on the fit method.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """

        logger.start_operation(
            f"Transforming data by dropping {len(self.dropped_columns)} zero variance columns."
        )
        try:
            X_transformed = X.drop(columns=self.dropped_columns, errors="ignore")
            logger.debug(
                f"Successfully dropped zero variance columns : {self.dropped_columns}"
            )
        except Exception as e:
            logger.error(f"Failed to transform VarianceFilter : {e}", exc_info=True)
            raise ValueError(f"Failed to transform VarianceFilter : {e}")
        finally:
            logger.end_operation()

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation("Fitting and transforming data with zero variance")
        try:
            X_transformed = self.fit(X).transform(X)
            logger.debug(
                f"Successfully fit_transformed zero variance columns : {self.dropped_columns}"
            )
        except Exception as e:
            logger.error(f"Failed to fit_transform VarianceFilter : {e}", exc_info=True)
            raise ValueError(f"Failed to fit_transform VarianceFilter : {e}")
        finally:
            logger.end_operation()
        return X_transformed

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "desc": f"Removes columns with zero variance. Dropped columns: {self.dropped_columns}",
        }
