import numpy as np
import pandas as pd

from ..utils.abstract import NonRequiredStep, Numerical
from ..utils.config import config
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class BinningTransformer(NonRequiredStep, Numerical):
    """
    Transformer for performing binning (using qcut) or equal-width binning (using cut)
    on continuous variables and replacing the values with numeric labels, but only if the number of unique values
    exceeds 50% of the number of samples in the column.


    Attributes:
        threshold (float) : percent of unique values in a column in order to classify for binning. Default : 0.5
        should_bin (dict) : dictionary to track which columns should be binned.
        bin_edges (dict) : dictionary to store the bin edged for each column.
    """

    PARAMS_GRID = {
        "binning_method": {"qcut", "cut"},
    }

    def __init__(self, binning_method: str = "qcut"):
        """
        Initializes the transformer with the number of bins for quantile binning and the binning method to use ('cut' or 'qcut').

        Args:
            n_bins (int): The number of bins to create (default is 4).
            binning_method (str): The binning method to use ('cut' for equal-width, 'qcut' for quantile binning) (default 'qcut').
        """

        self.n_bins = config.n_bins
        self.threshold = 0.5
        self.should_bin = {}  # A dictionary to track which columns should be binned
        self.bin_edges = {}  # Dictionary to store the bin edges for each column
        self.binning_method = binning_method

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "BinningTransformer":
        """
        Fits the transformer by calculating the bin edges for each continuous column if the number of unique values
        exceeds the threshold of 50%.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            BinningTransformer: The fitted transformer instance.
        """

        logger.start_operation(
            f"Fitting BinningTransformer with {self.n_bins} binning method : {self.binning_method}"
        )

        try:
            for column in X.select_dtypes(include=[np.number]).columns:
                unique_values_ratio = len(X[column].unique()) / len(X[column])
                if unique_values_ratio > self.threshold:
                    self.should_bin[column] = True

                    if self.binning_method == "cut":
                        logger.debug(
                            f"BinningTransformer: calculating bin edges for {column} using cut"
                        )
                        self.bin_edges[column] = np.linspace(
                            X[column].min(), X[column].max(), self.n_bins + 1
                        )
                    elif self.binning_method == "qcut":
                        logger.debug(
                            f"BinningTransformer: calculating bin edges for {column} using qcut"
                        )
                        self.bin_edges[column] = np.percentile(
                            X[column], np.linspace(0, 100, self.n_bins + 1)
                        )
                else:
                    self.should_bin[column] = False

            logger.debug("Successfully fitted BinningTransformer.")
        except Exception as e:
            logger.error(
                f"Failed to fit BinningTransformer with method {self.binning_method}: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Failed to fit BinningTransformer with method {self.binning_method}: {e}"
            )

        finally:
            logger.end_operation()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by replacing continuous values with their respective bin labels (numeric).

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data with bin labels.
        """

        logger.start_operation(
            f"Transforming BinningTransformer with {self.n_bins} and binning_method {self.binning_method}"
        )

        try:
            X_transformed = X.copy()
            for column in X_transformed.columns:
                if self.should_bin.get(column, False):

                    if self.binning_method == "cut":
                        logger.debug(
                            f"BinningTransformer: transforming column: {column} using cut"
                        )
                        X_transformed[column] = pd.cut(
                            X_transformed[column],
                            bins=self.bin_edges[column],
                            labels=False,
                            include_lowest=True,
                        )
                    elif self.binning_method == "qcut":
                        logger.debug(
                            f"BinningTransformer: transforming column : {column} using qcut"
                        )
                        X_transformed[column] = pd.qcut(
                            X_transformed[column],
                            q=self.n_bins,
                            labels=False,
                            duplicates="drop",
                        )

            logger.debug("Successfully transformed data with BinningTransformer")

        except Exception as e:
            logger.error(
                f"Failed to transform BinningTransformer with method {self.binning_method}: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Failed to transform BinningTransformer with method {self.binning_method}: {e}"
            )
        finally:
            logger.end_operation()
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data with bin labels.
        """

        logger.start_operation(
            f"Fitting ans transforming data with BinningTransformer n_bins: {self.n_bins} and binning_method: {self.binning_method}"
        )
        try:
            transformed_X = self.fit(X).transform(X)

            logger.debug(
                f"Successfully fit_transformed data data with BinningTransformer n_bins: {self.n_bins} and binning_method: {self.binning_method}"
            )

        except Exception as e:
            logger.error(
                f"Failed to fit_transform BinningTransformer with method {self.binning_method}: {e}",
                exc_info=True,
            )
            raise ValueError(
                f"Failed to fit_transform BinningTransformer with method {self.binning_method}: {e}"
            )
        finally:
            logger.end_operation()
        return transformed_X

    def is_numeric(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.

        Returns:
            dict: Description of the transformer.
        """
        return {
            "desc": "Performs binning on continuous variables and replaces them with numeric labels. ",
            "params": {"n_bins": self.n_bins, "binning_method": self.binning_method},
        }
