import pandas as pd

from ..utils.abstract import Categorical, RequiredStep
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class UniqueFilter(RequiredStep, Categorical):
    """
    Transformer to remove categorical columns 100% unique values.

    Attributes:
        dropped_columns (list): List of dropped columns.
    """

    def __init__(self):
        """
        Initializes the transformer with an empty list of dropped columns.
        """
        self.dropped_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "UniqueFilter":
        """
        Identifies categorical columns with 100% unique values.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            UniqueFilter: The fitted transformer instance.
        """
        logger.start_operation("Fitting UniqueFilter")
        try:
            cat_cols = X.select_dtypes(exclude="number")
            self.dropped_columns = [
                col for col in cat_cols if X[col].nunique() == len(X)
            ]
            logger.debug("Successfully fitted UniqueFilter")
        except Exception as e:
            logger.error(f"Failed to fit UniqueFilter : {e}", exc_info=True)
            raise ValueError(f"Failed to fit Uniquefilter {e}")
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the identified categorical columns with 100% unique values based on the fit method.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation(
            f"Transforming data UniqueFilter by dropping {len(self.dropped_columns)} columns with unique values"
        )
        try:
            X_transformed = X.drop(columns=self.dropped_columns, errors="ignore")
            logger.debug("Successfully transformed UniqueFilter")
        except Exception as e:
            logger.error(f"Failed to transform UniqueFilter : {e}", exc_info=True)
            raise ValueError(f"Failed to transform Uniquefilter {e}")
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
        logger.start_operation(
            "Fitting and transforming categorical data with 100% unique values"
        )
        try:
            X_transformed = self.fit(X).transform(X)
            logger.debug("Successfully fit_transformed UniqueFilter")
        except Exception as e:
            logger.error(f"Failed to fit_transform UniqueFilter : {e}", exc_info=True)
            raise ValueError(f"Failed to fit_transform Uniquefilter {e}")
        finally:
            logger.end_operation()
        return X_transformed

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "desc": f"Removes categorical columns with 100% unique values. Dropped columns: {self.dropped_columns}",
        }
