import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..utils.abstract import DimentionReducer
from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class PCADimentionReducer(DimentionReducer):
    """
    Combines data standardization and PCA with automatic selection of the number of components
    to preserve 95% of the variance.
    """

    def __init__(self):
        """
        Initializes the PCA object with additional parameters.
        """
        super().__init__()
        self.reducer = None  # PCA will be initialized in fit
        self.scaler = StandardScaler()
        self.n_components = None  # Will be determined in fit

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "PCADimentionReducer":
        """
        Fits the scaler and PCA to the data, determining the number of components
        to preserve 95% of the variance.

        Args:
            X (pd.DataFrame or np.ndarray): Input data.
            y (optional): Target values (ignored).

        Returns:
            PCADimentionReducer: The fitted transformer.
        """
        logger.start_operation(
            f"Fitting PCADimentionReducer to data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        try:
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)

            # Fit PCA to determine the number of components
            temp_pca = PCA()
            temp_pca.fit(X_scaled)
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumulative_variance >= 0.95) + 1

            # Initialize PCA with the determined number of components
            self.reducer = PCA(n_components=self.n_components)
            self.reducer.fit(X_scaled)

            logger.debug(f"Number of components selected: {self.n_components}")
        except Exception as e:
            logger.error(f"Error in PCADimentionReducer fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transforms the input data using the fitted scaler and PCA.

        Args:
            X (pd.DataFrame or np.ndarray): Input data.
            y (optional): Target values (ignored).

        Returns:
            np.ndarray: Transformed data.
        """
        logger.start_operation(
            f"Transforming data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        try:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled)
        except Exception as e:
            logger.error(f"Error in PCADimentionReducer transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return pd.DataFrame(self.reducer.transform(X_scaled))

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits the transformer to the data and then transforms it.

        Args:
            X (pd.DataFrame or np.ndarray): Input data.
            y (optional): Target values (ignored).

        Returns:
            np.ndarray: Transformed data.
        """
        logger.start_operation("Fitting and transforming data using StandarizeAndPCA.")
        try:
            self.fit(X, y)
        except Exception as e:
            logger.error(f"Error in PCADimentionReducer fit_transform: {e}")
            raise e
        finally:
            return self.transform(X)

    def to_tex(self) -> dict:
        return {
            "desc": "Combines data standardization and PCA with automatic selection of the number of components to preserve 95% of the variance.",
            "params": {"n_components": self.n_components},
        }


class VIFDimentionReducer(DimentionReducer):
    """
    Removes columns with high variance inflation factor (VIF > 10).
    """

    def __init__(self):
        """
        Initializes the VIFDimentionReducer.
        """
        self.multicollinear_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "VIFDimentionReducer":
        """
        Fits the VIFDimentionReducer to the data, identifying columns with high VIF.

        Args:
            X (pd.DataFrame): Input data.
            y (optional): Target values (ignored).

        Returns:
            VIFDimentionReducer: The fitted transformer.
        """
        logger.start_operation(
            f"Fitting VIF to data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        try:
            for col in X.columns:
                vif = variance_inflation_factor(X.values, X.columns.get_loc(col))
                if vif > 10:
                    self.multicollinear_columns.append(col)
            logger.debug(f"Columns with high VIF: {self.multicollinear_columns}")
        except Exception as e:
            logger.error(f"Error in VIFDimentionReducer fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Removes columns with high VIF from the data.

        Args:
            X (pd.DataFrame): Input data.
            y (optional): Target values (ignored).

        Returns:
            pd.DataFrame: Transformed data.
        """
        logger.start_operation("Transforming data.")
        try:
            X_copy = X.copy()
            X_copy.drop(columns=self.multicollinear_columns, inplace=True)
        except Exception as e:
            logger.error(f"Error in VIFDimentionReducer transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X_copy

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits the VIFDimentionReducer to the data and then transforms it.

        Args:
            X (pd.DataFrame): Input data.
            y (optional): Target values (ignored).

        Returns:
            pd.DataFrame: Transformed data.
        """
        logger.start_operation("Fitting and transforming data using VIF.")
        try:
            self.fit(X)
            logger.debug(
                f"Removing columns with high VIF: {self.multicollinear_columns}"
            )
        except Exception as e:
            logger.error(f"Error in VIFDimentionReducer fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return self.transform(X)

    def to_tex(self) -> dict:
        return {
            "desc": "Removes columns with high variance inflation factor (VIF > 10).",
        }


class UMAPDimentionReducer(DimentionReducer):
    """
    Reduces the dimensionality of the data using UMAP.
    """

    def __init__(self):
        self.reducer = None
        self.n_components = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "UMAPDimentionReducer":
        """
        Fits the UMAPDimentionReducer to the data.
        """
        logger.start_operation(
            f"Fitting UMAPDimentionReducer to data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        try:
            if X.shape[1] > 100:
                self.n_components = config.umap_components
            else:
                self.n_components = int(X.shape[1] / 2)
            self.reducer = umap.UMAP(n_components=self.n_components)
            self.reducer.fit(X)
            logger.debug(f"Number of components selected: {self.n_components}")
        except Exception as e:
            logger.error(f"Error in DimentionReducerUMAP fit: {e}")
            raise e
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transforms the input data using the fitted UMAP reducer.
        """
        logger.start_operation(
            f"Transforming data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        try:
            X = pd.DataFrame(self.reducer.transform(X))
        except Exception as e:
            logger.error(f"Error in DimentionReducerUMAP transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits the transformer to the data and then transforms it.
        """
        logger.start_operation(
            "Fitting and transforming data using DimentionReducerUMAP."
        )
        try:
            self.fit(X)
            X = self.transform(X)
            logger.debug(f"Reducing data to {self.n_components} components.")
        except Exception as e:
            logger.error(f"Error in DimentionReducerUMAP fit_transform: {e}")
            raise e
        finally:
            logger.end_operation()
        return X

    def to_tex(self) -> dict:
        return {
            "desc": "Reduces the dimensionality of the data using UMAP.",
            "params": {"n_components": self.n_components},
        }
