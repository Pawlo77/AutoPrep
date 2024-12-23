import pandas as pd
from ..utils.abstract import Numerical, RequiredStep, NonRequiredStep
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

class VarianceAndUniqueFilter(RequiredStep, Numerical):
    """
    Transformer to remove columns with zero variance or with at least unique_threshold% unique values.
    Default threshold is 90%.

    Attributes:
        unique_threshold (float): Percentage threshold for unique values. (0-100)
        dropped_columns (list): List of dropped columns.
    """

    def __init__(self, unique_threshold: float = 90.0):
        """
        Initializes the transformer with a specified threshold (%) for unique values.

        Args:
            unique_threshold (float): Percentage threshold for unique values.
        """
        if not (0 <= unique_threshold <= 100):
            raise ValueError("unique_threshold must be between 0 and 100.")
        self.unique_threshold = unique_threshold
        self.dropped_columns = []

    def fit(self, X: pd.DataFrame) -> "VarianceAndUniqueFilter":
        """
        Identifies columns with zero variance or with high percentage of unique values.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            VarianceAndUniqueFilter: The fitted transformer instance.
        """
        # Find columns with zero variance
        zero_variance = X.var() == 0

        # Find columns with high percentage of unique values
        high_unique = (X.nunique() / len(X)) * 100 >= self.unique_threshold

        self.dropped_columns = X.columns[zero_variance | high_unique].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the identified columns with zero variance or high percentage of unique values based on the fit method.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        return X.drop(columns=self.dropped_columns, errors='ignore')

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        return self.fit(X).transform(X)

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "name": "VarianceAndUniqueFilter",
            "desc": f"Removes columns with zero variance or >= {self.unique_threshold}% unique values. Dropped columns: {self.dropped_columns}",
            "params": {"unique_threshold": self.unique_threshold}
        }


class CorrelationFilter(RequiredStep,Numerical):
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
        logger.start_operation(f"Fitting CorrelationFilter with threshold {self.threshold}")
        corr_matrix = X.corr().abs()
        correlated_pairs = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.threshold:
                    correlated_pairs.add(corr_matrix.columns[j]) #only the second column of the pair is dropped

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

        X = X.drop(columns=self.dropped_columns, errors='ignore')

        if y is not None:
            y_name = y.name if y.name is not None else "y"
            logger.debug(f"Appending target variable '{y_name}' to the transformed data.")
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
        logger.start_operation(f"Fitting and transforming data with correlation threshold {self.threshold}")
        result = self.fit(X).transform(X, y)
        logger.end_operation()
        return result
    
    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "name": "CorrelationFilter",
            "desc": f"Removes one column from pairs of columns correlated above correlation threshold: {self.threshold}.",
            "params": {"threshold": self.threshold}
        }
    

class CorrelationSelector(NonRequiredStep, Numerical):
    """
       Transformer to select k% (rounded to whole number) of features that are most correlated with the target variable.

       Attributes:
            k (float): The percentage of top features to keep based on their correlation with the target.
            selected_columns (list): List of selected columns based on correlation with the target.
    """
    def __init__(self, k: float = 10.0):
        """
        Initializes the transformer with a specified percentage of top correlated features to keep.

        Args:
            k (float): The percentage of features to retain based on their correlation with the target.
        """
        if not (0 <= k <= 100):
            raise ValueError("k must be between 0 and 100.")
        self.k = k
        self.selected_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CorrelationSelector":
        """
        Identifies the top k% (rounded to whole value) of features most correlated with the target variable.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            CorrelationSelector: The fitted transformer instance.
        """
        logger.start_operation(f"Fitting CorrelationSelector with top {self.k}% correlated features.")

        corr_with_target = X.corrwith(y).abs()
        sorted_corr = corr_with_target.sort_values(ascending=False)
        num_top_features = max(1,round(np.ceil(len(sorted_corr) * self.k / 100)))
        self.selected_columns = sorted_corr.head(num_top_features).index.tolist()

        logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most correlated with the target variable.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% correlated features.
        """
        logger.start_operation(f"Transforming data by selecting {len(self.selected_columns)} most correlated features.")

        X_selected = X[self.selected_columns].copy()

        if y is not None:
            y_name = y.name if y.name is not None else "y"
            logger.debug(f"Appending target variable '{y_name}' to the transformed data.")
            X_selected[y_name] = y

        logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most correlated features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        logger.start_operation(f"Fitting and transforming data with top {self.k}% correlated features.")
        result = self.fit(X, y).transform(X, y)
        logger.end_operation()
        return result
    
    def is_numerical(self) -> bool:
        return True
    

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "name": "CorrelationSelector",
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most correlated with the target variable. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k}
        }
