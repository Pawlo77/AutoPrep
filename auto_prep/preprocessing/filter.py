import pandas as pd

from ..utils.abstract import Categorical, Numerical, RequiredStep
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

    def fit(self, X: pd.DataFrame) -> "VarianceFilter":
        """
        Identifies columns with zero variances and adds to dropped_columns list.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            VarianceAndUniqueFilter: The fitted transformer instance.
        """
        logger.start_operation("Fitting VarianceFilter")
        zero_variance = X.var() == 0
        self.dropped_columns = X.columns[zero_variance].tolist()
        logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        logger.end_operation()
        return X.drop(columns=self.dropped_columns, errors="ignore")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation("Fitting and transforming data with zero variance")
        logger.end_operation()
        return self.fit(X).transform(X)

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "name": "VarianceFilter",
            "desc": f"Removes columns with zero variance. Dropped columns: {self.dropped_columns}",
            "params": {},
        }


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

    def fit(self, X: pd.DataFrame) -> "UniqueFilter":
        """
        Identifies categorical columns with 100% unique values.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            UniqueFilter: The fitted transformer instance.
        """
        logger.start_operation("Fitting UniqueFilter")
        # Select only categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"])
        self.dropped_columns = [
            col for col in cat_cols.columns if X[col].nunique() == len(X)
        ]
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
        logger.end_operation()
        return X.drop(columns=self.dropped_columns, errors="ignore")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        logger.end_operation()
        return self.fit(X).transform(X)

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "name": "UniqueFilter",
            "desc": f"Removes categorical columns with 100% unique values. Dropped columns: {self.dropped_columns}",
            "params": {},
        }


class CorrelationFilter(RequiredStep, Numerical):
    """
    Transformer to detect highly correlated features and drop one of them. Pearsons correlation is used.
    Is a required step in preprocessing.

    Attributes:
        threshold (float): Correlation threshold above which features are considered highly correlated.
        dropped_columns (list): List of columns that were dropped due to high correlation.
    """

    def __init__(self, threshold: float = 0.8):
        """
        Initializes the filter with a specified correlation threshold.

        Args:
            threshold (float): Correlation threshold above which features are considered highly correlated.
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.threshold = threshold
        self.dropped_columns = []

    def fit(self, X: pd.DataFrame) -> "CorrelationFilter":
        """
        Identifies highly correlated features. Adds the second one from the pair to the list of columns to be dropped.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            CorrelationFilter: The fitted filter instance.
        """
        logger.start_operation(
            f"Fitting CorrelationFilter with threshold {self.threshold}"
        )
        corr_matrix = X.corr().abs()
        correlated_pairs = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.threshold:
                    correlated_pairs.add(
                        corr_matrix.columns[j]
                    )  # only the second column of the pair is dropped

        self.dropped_columns = list(correlated_pairs)
        logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Drops all features identified as highly correlated with another feature.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with correlated columns removed.
        """
        logger.start_operation(
            f"Transforming data by dropping {len(self.dropped_columns)} highly correlated columns."
        )

        X = X.drop(columns=self.dropped_columns, errors="ignore")

        if y is not None:
            y_name = y.name if y.name is not None else "y"
            logger.debug(
                f"Appending target variable '{y_name}' to the transformed data."
            )
            X[y_name] = y

        logger.end_operation()
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits and transforms the data by removing correlated features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data.
        """
        logger.start_operation(
            f"Fitting and transforming data with correlation threshold {self.threshold}"
        )
        result = self.fit(X).transform(X, y)
        logger.end_operation()
        return result

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "name": "CorrelationFilter",
            "desc": f"Removes one column from pairs of columns correlated above correlation threshold: {self.threshold}.",
            "params": {"threshold": self.threshold},
        }
