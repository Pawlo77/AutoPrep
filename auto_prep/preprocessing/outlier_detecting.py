import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from statsmodels.regression.linear_model import OLS

from ..utils.abstract import Numerical, RequiredStep
from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class OutlierDetector(RequiredStep, Numerical):
    """
    Performs Numerical data outlier detection
    """

    def __init__(self):
        """
        Args:
            method: The method to use for outlier detection.
        """

        self.method = config.outlier_detector_method

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "OutlierDetector":
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
        try:
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) == 0:
                raise ValueError("Non numerical columns found in input data.")
        except Exception as e:
            logger.error(f"Error in Outlier Detection fit: {e}")
            raise e
        finally:
            logger.end_operation()

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Applies cleaning and transformation operations to the input data.

        Args:
            X (pd.DataFrame): The input DataFrame to be cleaned and transformed.
            y (pd.Series): The target data.

        Returns:
            pd.DataFrame: The cleaned and transformed DataFrame.
        """
        logger.start_operation("Transforming data for outlier detection.")
        try:
            X = X.copy()

            if self.method == "cooks_distance":
                outliers = self._cooks_distance_outliers(X, y)
            elif self.method == "isolation_forest":
                outliers = self._isolation_forest_outliers(X)
            elif self.method == "iqr":
                outliers = self._iqr_outliers(X)
            else:
                outliers = self._zscore_outliers(X)

            logger.debug(f"Found {len(outliers[0])} outliers.")
            logger.critical(outliers)

            X = X.drop(outliers[0])
            logger.end_operation()
        except Exception as e:
            logger.error(f"Error in Outlier Detection: {e}")
            raise e
        finally:
            logger.end_operation()

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit and transform the data in one step.
        Args:
            X (pd.DataFrame): Input data
            y (pd.Series): Target data
        Returns:
            pd.DataFrame: Transformed data
        """
        logger.start_operation(
            f"Fitting ans transforming data with OutlierDetector method: {self.method}"
        )
        try:
            transformed_X = self.fit(X).transform(X)

            logger.debug(
                f"Successfully fit_transformed data data with OutlierDetector method: {self.method}"
            )

        except Exception as e:
            logger.error(
                f"Failed to fit_transform OutlierDetector method: {self.method}: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Failed to fit_transform OutlierDetector method: {self.method}"
            ) from e
        finally:
            logger.end_operation()
        return transformed_X

    def _zscore_outliers(self, X: pd.DataFrame) -> tuple:
        """
        Detect outliers using Z-score method
        Args:
            X: Input data
        Returns:
            Tuple of arrays containing row and column indices of outliers
        """

        logger.start_operation("Detecting outliers using Z-score.")
        try:
            threshold = config.outlier_detector_settings["zscore_threshold"]
            z_scores = np.abs(stats.zscore(X, axis=0))
        except Exception as e:
            logger.error(f"Error in Z-score outlier detection: {e}")
            raise e
        finally:
            logger.end_operation()
        return np.where(z_scores > threshold)

    def _iqr_outliers(self, X: pd.DataFrame) -> tuple:
        """
        Detect outliers using IQR method
        Args:
            X: Input data
        Returns:
            Tuple of arrays containing row and column indices of outliers
        """
        logger.start_operation("Detecting outliers using IQR.")
        try:
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
        except Exception as e:
            logger.error(f"Error in IQR outlier detection: {e}")
            raise e
        finally:
            logger.end_operation()
        return np.where((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))

    def _isolation_forest_outliers(self, X: pd.DataFrame) -> tuple:
        """
        Detect outliers using Isolation Forest method
        Args:
            X: Input data
        Returns:
            Tuple of arrays containing row and column indices of outliers
        """
        logger.start_operation("Detecting outliers using Isolation Forest.")
        try:
            n_estimators = config.outlier_detector_settings["isol_forest_n_estimators"]
            clf = IsolationForest(n_estimators=n_estimators)
            clf.fit(X)
            outliers = np.where(clf.predict(X) == -1)
            logger.debug(f"Found {len(outliers)} outliers.")
        except Exception as e:
            logger.error(f"Error in Isolation Forest outlier detection: {e}")
            raise e
        finally:
            logger.end_operation()
        return outliers

    def _cooks_distance_outliers(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Detect outliers using Cook's Distance method
        Args:
            X: Input data
            y: Target data
        Returns:
            Tuple of arrays containing row and column indices of outliers
        """
        logger.start_operation("Detecting outliers using Cook's Distance.")
        try:
            model = OLS(y, X).fit()
            infl = model.get_influence()
            cooks_distance, _ = infl.cooks_distance
            threshold = config.outlier_detector_settings["cooks_distance_threshold"]
        except Exception as e:
            logger.error(f"Error in Cook's Distance outlier detection: {e}")
            raise e
        finally:
            logger.end_operation()
        return np.where(cooks_distance > threshold)

    def to_tex(self) -> dict:
        return {
            "desc": "Detects outliers in numerical data using specified method.",
            "params": {"method": self.method},
        }
