from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..utils.abstract import NonRequiredStep
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class DimentionReducer(NonRequiredStep, ABC):
    """
    Abstract class for dimensionality reduction techniques.
    """

    def __init__(self):
        super().__init__()
        self.reducer = None

    @abstractmethod
    def fit(self, X, y=None) -> "DimentionReducer":
        pass

    @abstractmethod
    def transform(self, X, y=None) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, X, y=None) -> pd.DataFrame:
        pass

    @abstractmethod
    def is_applicable(X):
        pass

    @abstractmethod
    def is_numerical(self) -> bool:
        pass

    @abstractmethod
    def to_tex(self) -> dict:
        pass


class PCADimentionReducer(DimentionReducer):
    """
    Combines data standardization and PCA with automatic selection of the number of components
    to preserve 95% of the variance.
    """

    def __init__(self, **kwargs):
        """
        Initializes the PCA object with additional parameters.
        """
        super().__init__()
        self.reducer = None  # PCA will be initialized in fit
        self.scaler = StandardScaler()
        self.n_components = None  # Will be determined in fit
        self.pca_kwargs = kwargs

    def fit(self, X, y=None) -> "PCADimentionReducer":
        """
        Fits the scaler and PCA to the data, determining the number of components
        to preserve 95% of the variance.

        Args:
            X (pd.DataFrame or np.ndarray): Input data.
            y (optional): Target values (ignored).

        Returns:
            StandarizeAndPCA: The fitted transformer.
        """
        logger.start_operation(
            f"Fitting StandarizeAndPCA to data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Fit PCA to determine the number of components
        temp_pca = PCA(**self.pca_kwargs)
        temp_pca.fit(X_scaled)
        cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
        self.n_components = np.argmax(cumulative_variance >= 0.95) + 1

        # Initialize PCA with the determined number of components
        self.reducer = PCA(n_components=self.n_components, **self.pca_kwargs)
        self.reducer.fit(X_scaled)

        logger.debug(f"Number of components selected: {self.n_components}")
        logger.end_operation()

        return self

    def transform(self, X, y=None) -> pd.DataFrame:
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
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled)
        logger.end_operation()
        return pd.DataFrame(self.reducer.transform(X_scaled))

    def fit_transform(self, X, y=None) -> pd.DataFrame:
        """
        Fits the transformer to the data and then transforms it.

        Args:
            X (pd.DataFrame or np.ndarray): Input data.
            y (optional): Target values (ignored).

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)

    def is_applicable(X):
        return np.shape(X)[0] > 1 and np.shape(X)[1] > 1

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        return {"name": "StandarizeAndPCA", "desc": "", "params": {}}


class VIFDimentionReducer(DimentionReducer):
    """
    Removes columns with high variance inflation factor (VIF > 10).
    """

    def __init__(self):
        """
        Initializes the VIFDimentionReducer.
        """
        self.multicollinear_columns = []

    def fit(self, X: pd.DataFrame, y=None) -> "VIFDimentionReducer":
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
        for col in X.columns:
            vif = variance_inflation_factor(X.values, X.columns.get_loc(col))
            if vif > 10:
                self.multicollinear_columns.append(col)
        logger.debug(f"Columns with high VIF: {self.multicollinear_columns}")
        logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Removes columns with high VIF from the data.

        Args:
            X (pd.DataFrame): Input data.
            y (optional): Target values (ignored).

        Returns:
            pd.DataFrame: Transformed data.
        """
        return X.drop(columns=self.multicollinear_columns)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fits the VIFDimentionReducer to the data and then transforms it.

        Args:
            X (pd.DataFrame): Input data.
            y (optional): Target values (ignored).

        Returns:
            pd.DataFrame: Transformed data.
        """
        logger.start_operation("Fitting and transforming data using VIF.")
        self.fit(X)
        logger.debug(f"Removing columns with high VIF: {self.multicollinear_columns}")
        logger.end_operation()
        return self.transform(X)

    def is_applicable(X):
        return np.shape(X)[0] > 1 and np.shape(X)[1] > 1

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        return {
            "name": "VIF",
            "desc": "Removes columns with high variance inflation factor (VIF > 10).",
            "params": {},
        }


class UMAPDimentionReducer(DimentionReducer):
    """
    Reduces the dimensionality of the data using UMAP.
    """

    def __init__(self):
        self.reducer = None
        self.n_components = None

    def fit(self, X, y=None) -> "UMAPDimentionReducer":
        """
        Fits the UMAPDimentionReducer to the data.
        """
        logger.start_operation(
            f"Fitting UMAPDimentionReducer to data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        if X.shape[1] > 100:
            self.n_components = 50
        else:
            self.n_components = int(X.shape[1] / 2)
        self.reducer = umap.UMAP(n_components=self.n_components)
        self.reducer.fit(X)
        logger.debug(f"Number of components selected: {self.n_components}")
        logger.end_operation()
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """
        Transforms the input data using the fitted UMAP reducer.
        """
        return pd.DataFrame(self.reducer.transform(X))

    def fit_transform(self, X, y=None) -> pd.DataFrame:
        """
        Fits the transformer to the data and then transforms it.
        """
        logger.start_operation(
            "Fitting and transforming data using DimentionReducerUMAP."
        )
        self.fit(X)
        logger.end_operation()
        return self.transform(X)

    def is_applicable(X):
        return np.shape(X)[0] > 1 and np.shape(X)[1] > 1

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        return {
            "name": "DimentionReducerUMAP",
            "desc": "Reduces the dimensionality of the data using UMAP.",
            "params": {"n_components": self.n_components},
        }
