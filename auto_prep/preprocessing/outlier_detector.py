import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from statsmodels.regression.linear_model import OLS
from utils.abstract import Numerical, RequiredStep
from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class OutlierDetector(RequiredStep, Numerical):
    """
    Performs Numerical data outlier detection
    """

    def __init__(self, method, **kwargs):
        """
        Args:
            method: The method to use for outlier detection
            kwargs: Additional parameters for specific methods
        """
        self.available_methods = {
            "zscore": self._zscore_outliers,
            "iqr": self._iqr_outliers,
            "isolation_forest": self._isolation_forest_outliers,
            "cooks_distance": self._cooks_distance_outliers,
        }
        if method not in self.available_methods:
            raise ValueError(
                f"Method {method} not supported. Supported methods are "
                f"{self.available_methods.keys()}"
            )
        self.method = method
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierDetector":
        """Identify feature types in the dataset.

        Args:
            X (pd.DataFrame): Input features.
            y: Ignored. Exists for scikit-learn compatibility.

        Returns:
            OutlierDetector: Fitted transformer.
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
            outliers = self.available_methods[self.method](X)
            logger.debug(f"Found {len(outliers)} outliers.")
            X = X.drop(outliers[0])
            logger.end_operation()
        except Exception as e:
            logger.error(f"Error in Outlier Detection: {e}")
            raise e
        return X

    def _zscore_outliers(self, X):
        """
        Detect outliers using Z-score method
        Args:
            X: Input data
        Returns:
            Tuple of arrays containing row and column indices of outliers
        """
        threshold = self.kwargs.get("threshold", 3)
        z_scores = np.abs(stats.zscore(X, axis=0))
        return np.where(z_scores > threshold)

    def _iqr_outliers(self, X):
        """
        Detect outliers using IQR method
        Args:
            X: Input data
        Returns:
            Tuple of arrays containing row and column indices of outliers
        """
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        return np.where((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))

    def _isolation_forest_outliers(self, X):
        """
        Detect outliers using Isolation Forest method
        Args:
            X: Input data
        Returns:
            Tuple of arrays containing row and column indices of outliers
        """
        n_estimators = self.kwargs.get("n_estimators", 100)
        clf = IsolationForest(n_estimators=n_estimators)
        clf.fit(X)
        return np.where(clf.predict(X) == -1)

    def _cooks_distance_outliers(self, X, y):
        """
        Detect outliers using Cook's Distance method
        Args:
            X: Input data
            y: Target data
        Returns:
            Tuple of arrays containing row and column indices of outliers
        """
        model = OLS(y, X).fit()
        infl = model.get_influence()
        cooks_distance, _ = infl.cooks_distance
        threshold = self.kwargs.get("threshold", 4 / X.shape[0])
        return np.where(cooks_distance > threshold)

    def is_numerical(self) -> bool:
        return False

    def to_tex(self) -> dict:
        return {
            "name": "OutlierDetector",
            "desc": "Detects outliers in numerical data using specified method.",
            "params": {"method": self.method, "kwargs": self.kwargs},
        }
