import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..utils.abstract import Numerical, RequiredStep
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ColumnScaler(RequiredStep, Numerical):
    """
    Scaler for all numerical features. This class applies scaling technique based on users choice to
    all numerical features.

    Available scaling methods: MinMaxScaler, StandardScaler, RobustScaler from sklearn.

    Attributes:
        scaler (object): fitted scaler instance.
    """

    PARAMS_GRID = {
        "method": ["standard", "minmax", "robust"],
    }

    def __init__(self, method: str = "standard"):
        """
        Initializes the scaler with the specified scaling type. Default : StandardScaler

        Args:
            method (str): The type of scaler to use ('minmax', 'standard', or 'robust').
        """

        self.method = method

        if self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        else:

            raise ValueError(
                "Invalid scaler_type. Choose from : 'minmax', 'standard', 'robust'."
            )

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "ColumnScaler":
        """
        Fits the chosen scaler to the numerical features in the data.

        Args:
            X (pd.DataFrame): The feature data to fit the scaler to.

        Returns:
            ColumnScaler: The fitted scaler instance.
        """
        logger.start_operation(
            f"Fitting ColumnScaler with type '{self.method}' to data with {X.shape[0]} rows and {X.shape[1]} columns."
        )
        try:

            numerical_cols = X.select_dtypes(include=["number"]).columns
            if numerical_cols.empty:
                raise ValueError("Scaler: No numerical columns found in the dataset.")
            self.scaler.fit(X[numerical_cols])
            logger.debug(f"Successfully fitted ColumnScaler with method {self.method}")

        except Exception as e:
            logger.error(f"Failed to fit ColumnScaler: {e}", exc_info=True)
            raise ValueError(f"An error occurred while fitting ColumnScaler: {e}")
        finally:
            logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transforms numeric feature data using the fitted scaler.

        Args:
            X (pd.DataFrame): The feature data to transform.
            y (pd.Series, optional): The target variable.

        Returns:
            pd.DataFrame: The transformed feature data.
        """
        logger.start_operation(
            f"Scaler: Transforming data with {X.shape[0]} rows and {X.shape[1]} columns."
        )

        try:
            X_transformed = X.copy()

            numerical_cols = X_transformed.select_dtypes(include=["number"]).columns
            X_transformed[numerical_cols] = self.scaler.transform(
                X_transformed[numerical_cols]
            )
            logger.debug(
                f"Successfully transformed ColumnScaler with method {self.method}"
            )

        except Exception as e:
            logger.error(f"Failed to transform ColumnScaler {e}", exc_info=True)
            raise ValueError(f"An error occurred while transforming ColumnScaler: {e}")

        finally:
            logger.end_operation()

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits and transforms the feature data using the chosen scaler.

        Args:
            X (pd.DataFrame): The feature data to transform.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed feature data.
        """

        logger.start_operation(
            f"Fitting and transforming data using '{self.method}' scaler."
        )
        try:
            result = self.fit(X).transform(X, y)
            logger.debug(
                f"Successfully fit_transformed data with ColumnScaler method : {self.method}"
            )

        except Exception as e:
            logger.error(f"Failed to fit_transform ColumnScaler {e}", exc_info=True)
            raise ValueError(f"An error occurred while fit_transform ColumnScaler: {e}")

        finally:
            logger.end_operation()

        return result

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        This method returns a short description of the Scaler that was used in a form of dictionary.


        """
        return {
            "desc": "Scales numerical columns using one of 3 scaling methods.",
            "params": {"method": self.method},
        }
