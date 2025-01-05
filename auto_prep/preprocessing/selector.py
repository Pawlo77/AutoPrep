import numpy as np
import pandas as pd
<<<<<<< HEAD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa F401

from ..utils.abstract import (
    Categorical,
    FeatureImportanceSelector,
    NonRequiredStep,
    Numerical,
    RequiredStep,
)
=======
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..utils.abstract import Categorical, NonRequiredStep, Numerical, RequiredStep
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1
from ..utils.logging_config import setup_logger
from ..utils.config import config

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
<<<<<<< HEAD
=======

>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1
        self.dropped_columns = []

    def fit(self, X: pd.DataFrame) -> "VarianceFilter":
        """
        Identifies columns with zero variances and adds to dropped_columns list.

        Args:
            X (pd.DataFrame): The input feature data.

        Returns:
            VarianceFilter: The fitted transformer instance.
        """
        logger.start_operation("Fitting VarianceFilter")
        try:
            zero_variance = X.var() == 0
            self.dropped_columns = X.columns[zero_variance].tolist()
            logger.debug(f'Successfully fitted VarianceFilter')
            return self
        
        except Exception as e:
            logger.error(f'Failed to fit VarianceFilter {e}', exc_info=True)
            raise ValueError(f'Failed to fit VarianceFilter {e}')
        
        finally:
            logger.end_operation()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the identified columns with zero variance based on the fit method.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation(
            f"Transforming data with Variance Filter by dropping {len(self.dropped_columns)} zero variance columns."
        )
<<<<<<< HEAD
        try:
            transformed_X= X.drop(columns=self.dropped_columns, errors="ignore")
            logger.debug(f"Successfully transformed data with VarianceFilter. Final shape: {transformed_X.shape}")
            return transformed_X
        
        except Exception as e:
            logger.error(f'Failed to transform VarianceFilter {e}', exc_info=True)
            raise ValueError(f'Failed to transform VarianceFilter {e}')
        
        finally:
            logger.end_operation()
=======
        logger.end_operation()
        return X.drop(columns=self.dropped_columns, errors="ignore")
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation("Fitting and transforming data with VarianceFilter")
        try:
            transformed_X = self.fit(X).transform(X)
            logger.debug(f"Successfully fit_transformed data with VarianceFilter. Final shape: {transformed_X.shape}")
            return transformed_X
        
        except Exception as e:
            logger.error(f'Failed to fit_transform VarianceFilter {e}', exc_info=True)
            raise ValueError(f'Failed to fit_transform VarianceFilter {e}')
        
        finally:
            logger.end_operation()

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "desc": f"Removes columns with zero variance. Dropped columns: {self.dropped_columns}",
<<<<<<< HEAD
=======
            "params": {},
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1
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
<<<<<<< HEAD
        try:
            # Select only categorical columns
            cat_cols = X.select_dtypes(include=["object", "category"])
            self.dropped_columns = [
                col for col in cat_cols.columns if X[col].nunique() == len(X)
            ]
            logger.debug('Successfully fitted UniqueFilter')
            return self
        
        except Exception as e:
            logger.error(f'Failed to fit UniqueFilter : {e}', exc_info=True)
            raise ValueError(f'Failed to fit UniqueFilter: {e}')
        
        finally:
            logger.end_operation()
=======
        # Select only categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"])
        self.dropped_columns = [
            col for col in cat_cols.columns if X[col].nunique() == len(X)
        ]
        logger.end_operation()
        return self
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

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
<<<<<<< HEAD
        try:
            transformed_X= X.drop(columns=self.dropped_columns, errors="ignore")
            logger.debug(f"Successfully transformed UniqueFilter")
            return transformed_X
        
        except Exception as e:
            logger.error(f'Failed to transform UniqueFilter: {e}', exc_info=True)
            raise ValueError(f"Failed to transform UniqueFilter: {e}")
        
        finally:
            logger.end_operation()
=======
        logger.end_operation()
        return X.drop(columns=self.dropped_columns, errors="ignore")
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fits and transforms the data in one step.

        Args:
            X (pd.DataFrame): The feature data.

        Returns:
            pd.DataFrame: The transformed data without dropped columns.
        """
        logger.start_operation(
<<<<<<< HEAD
            "Fitting and transforming categorical data with 100% unique values"
        )
        try:
            transformed_X= self.fit(X).transform(X)
            logger.debug(f"Successfully fit_transformed UniqueFilter")
            return transformed_X
        
        except Exception as e:
            logger.error(f'Failed to fit_transform UniqueFilter: {e}', exc_info=True)
            raise ValueError(f"Failed to fit_transform UniqueFilter: {e}")
        
        finally:
            logger.end_operation()
=======
            f"Fitting and transforming categorical data with 100% unique values"
        )
        logger.end_operation()
        return self.fit(X).transform(X)
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

    def is_numerical(self) -> bool:
        return False

    def to_tex(self) -> dict:
        """
        Returns a description of the transformer in dictionary format.
        """
        return {
            "desc": f"Removes categorical columns with 100% unique values. Dropped columns: {self.dropped_columns}",
<<<<<<< HEAD
=======
            "params": {},
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1
        }


class CorrelationFilter(RequiredStep, Numerical):
    """
    Transformer to detect highly correlated features and drop one of them. Pearsons correlation is used.
    Is a required step in preprocessing.

    Attributes:
<<<<<<< HEAD
=======
        threshold (float): Correlation threshold above which features are considered highly correlated.
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1
        dropped_columns (list): List of columns that were dropped due to high correlation.
    """

    def __init__(self):
        """
        Initializes the filter with a specified correlation threshold.

        Args:
            correlation_threshold (float): Correlation threshold above which features are considered highly correlated.
        """
        self.correlation_threshold = config.correlation_threshold
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
<<<<<<< HEAD
            f"Fitting CorrelationFilter with threshold {self.correlation_threshold}"
        )
        try:
            corr_matrix = X.corr().abs()
            correlated_pairs = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.threshold:
                        correlated_pairs.add(
                            corr_matrix.columns[j]
                        )  # only the second column of the pair is dropped
=======
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
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

            self.dropped_columns = list(correlated_pairs)
            logger.debug(f'Successfully fitted CorrelationFilter with threshold: {self.correlation_threshold}')
            return self
        
        except Exception as e:
            logger.error(f'Failed to fit Correlation fiter with threshold: {self.correlation_threshold}. : {e}', exc_info=True)
            raise ValueError(f'Failed to fit Correlation fiter with threshold: {self.correlation_threshold}. : {e}')
        
        finally:
            logger.end_operation()

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

<<<<<<< HEAD
        try:
            X = X.drop(columns=self.dropped_columns, errors="ignore")
            if y is not None:
                y_name = y.name if y.name is not None else "y"
                logger.debug(
                    f"Appending target variable '{y_name}' to the transformed data."
                )
                X[y_name] = y
            logger.debug(f'Successfully transformed CorrelationFilter')
            return X
        
        except Exception as e:
            logger.error(f'Failed to transform CorrelationFilter : {e}', exc_info=True)
            raise ValueError(f'Failed to transform CorrelationFilter : {e}')
        
        finally:
            logger.end_operation()
=======
        X = X.drop(columns=self.dropped_columns, errors="ignore")

        if y is not None:
            y_name = y.name if y.name is not None else "y"
            logger.debug(
                f"Appending target variable '{y_name}' to the transformed data."
            )
            X[y_name] = y

        logger.end_operation()
        return X
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

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
<<<<<<< HEAD
        try:
            result = self.fit(X).transform(X, y)
            logger.debug('Successfully fit_transformed CorrelationFilter')
            return result
        
        except Exception as e:
            logger.error(f'Failed to fit_transform CorrelationFilter : {e}', exc_info=True)
            raise ValueError(f'Failed to fit_transform CorrelationFilter : {e}')
        
        finally:
            logger.end_operation()

=======
        result = self.fit(X).transform(X, y)
        logger.end_operation()
        return result
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
<<<<<<< HEAD
            "desc": f"Removes one column from pairs of columns correlated above correlation threshold.",
            "params": {"threshold": self.correlation_threshold},
=======
            "name": "CorrelationFilter",
            "desc": f"Removes one column from pairs of columns correlated above correlation threshold: {self.threshold}.",
            "params": {"threshold": self.threshold},
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1
        }


class CorrelationSelector(NonRequiredStep, Numerical):
    """
<<<<<<< HEAD
    Transformer to select correlation_percent% (rounded to whole number) of features that are most correlated with the target variable.

    Attributes:
         selected_columns (list): List of selected columns based on correlation with the target.
    """

    def __init__(self):
=======
    Transformer to select k% (rounded to whole number) of features that are most correlated with the target variable.

    Attributes:
         k (float): The percentage of top features to keep based on their correlation with the target.
         selected_columns (list): List of selected columns based on correlation with the target.
    """

    def __init__(self, k: float = 10.0):
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1
        """
        Initializes the transformer with a specified percentage of top correlated features to keep.

        Args:
            correlation_percent (float): The percentage of features to retain based on their correlation with the target.
        """
        self.correlation_percent = config.correlation_percent
        self.selected_columns = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CorrelationSelector":
        """
        Identifies the top correlation_percent% (rounded to whole value) of features most correlated with the target variable.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            CorrelationSelector: The fitted transformer instance.
        """
        logger.start_operation(
<<<<<<< HEAD
            f"Fitting CorrelationSelector with top {self.correlation_percent}% correlated features."
        )
        try:
            corr_with_target = X.corrwith(y).abs()
            sorted_corr = corr_with_target.sort_values(ascending=False)
            num_top_features = max(1, round(np.ceil(len(sorted_corr) * self.correlation_percent / 100)))
            self.selected_columns = sorted_corr.head(num_top_features).index.tolist()
            logger.debug(f'Successfully fitted CorrelationSelector with {self.correlation_percent}% features')
            return self
        except Exception as e:
            logger.error(f'Failed to fit CorrelationSelector with {self.correlation_percent}: {e}', exc_info=True)
            raise ValueError(f'Failed to fit CorrelationSelector with {self.correlation_percent}')
        finally:
            logger.end_operation()
=======
            f"Fitting CorrelationSelector with top {self.k}% correlated features."
        )

        corr_with_target = X.corrwith(y).abs()
        sorted_corr = corr_with_target.sort_values(ascending=False)
        num_top_features = max(1, round(np.ceil(len(sorted_corr) * self.k / 100)))
        self.selected_columns = sorted_corr.head(num_top_features).index.tolist()

        logger.end_operation()
        return self
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top correlation_percent% of features most correlated with the target variable.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% correlated features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most correlated features."
        )
<<<<<<< HEAD
        try:
=======
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

            X_selected = X[self.selected_columns].copy()
            if y is not None:
                y_name = y.name if y.name is not None else "y"
                logger.debug(
                    f"Appending target variable '{y_name}' to the transformed data."
                )
                X_selected[y_name] = y

<<<<<<< HEAD
            logger.debug('Successfully transformed data with CorrelationSelector')
            return X_selected
        except Exception as e:
            logger.error(f'Failed to transform {X} with CorrelationSelector threshold {self.correlation_percent}: {e}', exc_info=True)
            raise ValueError(f'Failed to transform {X} with CorrelationSelector threshold {self.correlation_percent}: {e}')
        finally:
            logger.end_operation()
=======
        if y is not None:
            y_name = y.name if y.name is not None else "y"
            logger.debug(
                f"Appending target variable '{y_name}' to the transformed data."
            )
            X_selected[y_name] = y

        logger.end_operation()
        return X_selected
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most correlated features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% correlated features."
        )
<<<<<<< HEAD
        try:
            result = self.fit(X, y).transform(X, y)
            logger.debug('Successfully fit_transformed data with CorrelationSelector')
            return result
        except Exception as e:
            logger.error(f'Failed to fit_transform {X} with CorrelationSelector threshold {self.correlation_percent}: {e}', exc_info=True)
            raise ValueError(f'Failed to fit_transform {X} with CorrelationSelector threshold {self.correlation_percent}: {e}')
        finally:
            logger.end_operation()
=======
        result = self.fit(X, y).transform(X, y)
        logger.end_operation()
        return result
>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1

    def is_numerical(self) -> bool:
        return True

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
<<<<<<< HEAD
            "desc": f"Selects the top k% (rounded to whole number) of features most correlated with the target variable. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"correlation_percent": self.correlation_percent},
        }


=======
            "name": "CorrelationSelector",
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most correlated with the target variable. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k},
        }


class FeatureImportanceSelector(NonRequiredStep):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self, k: float = 10.0):
        """
        Initializes the transformer with a specified model and percentage of top important features to keep.

        Args:
            k (float): The percentage of features to retain based on their importance.
        """
        if not (0 <= k <= 100):
            raise ValueError("k must be between 0 and 100.")
        self.k = k
        self.selected_columns = []

    def fit(self, X, y):
        """
        Identifies the top k% (rounded to whole value) of features most important according to the model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceSelector: The fitted transformer instance.
        """
        pass

    def transform(self, X, y=None):
        """
        Selects the top k% of features most important according to the model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        pass

    def fit_transform(self, X, y):
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def is_applicable(self, dt):
        """
        Args:
            dt (pd.Series) - column that is considered to be preprocessed
                by that transformer.

        Returns:
            bool - True if it is possible to use this transofmation
                on passed data.
        """
        return True


>>>>>>> 977a1a4cde26cba2552ca81f213401cdd6ac27b1
class FeatureImportanceClassificationSelector(FeatureImportanceSelector):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model for classification.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self, k: float = 10.0):
        """
        Initializes the transformer with a specified percentage of top important features to keep.

        Args:
            k (float): The percentage of features to retain based on their importance.
        """
        super().__init__(k)
        self.feature_importances_ = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "FeatureImportanceClassificationSelector":
        """
        Identifies the feature importances according to the Random Forest model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceClassificationSelector: The fitted transformer instance.
        """
        logger.start_operation(
            f"Fitting FeatureImportanceClassificationSelector with top {self.k}% important features."
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        self.feature_importances_ = model.feature_importances_
        total_features = len(self.feature_importances_)
        num_features_to_select = int(np.ceil(total_features * self.k / 100))
        if num_features_to_select == 0:
            num_features_to_select = 1
        indices = np.argsort(self.feature_importances_)[-num_features_to_select:][::-1]
        self.selected_columns = X.columns[indices].tolist()

        logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most important according to the Random Forest model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most important features."
        )

        X_selected = X[self.selected_columns].copy()
        if y is not None:
            if isinstance(y, list):
                y = pd.Series(y)
            y_name = y.name if y.name is not None else "y"
            logger.debug(
                f"Appending target variable '{y_name}' to the transformed data."
            )
            X_selected[y_name] = y.values

        logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.
        """

        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% important features."
        )
        result = self.fit(X, y).transform(X, y)
        logger.end_operation()
        return result

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "name": "FeatureImportanceClassificationSelector",
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most important according to Random Forest model for classification. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k},
        }

    def is_numerical(self) -> bool:
        return False


class FeatureImportanceRegressionSelector(FeatureImportanceSelector):
    """
    Transformer to select k% (rounded to whole number) of features
    that are most important according to Random Forest model for regression.

    Attributes:
        k (float): The percentage of top features to keep based on their importance.
        selected_columns (list): List of selected columns based on feature importance.
    """

    def __init__(self, k: float = 10.0):
        """
        Initializes the transformer with a specified percentage of top important features to keep.

        Args:
            k (float): The percentage of features to retain based on their importance.
        """
        super().__init__(k)
        self.feature_importances_ = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "FeatureImportanceRegressionSelector":
        """
        Identifies the feature importances according to the Random Forest model.

        Args:
            X (pd.DataFrame): The input feature data.
            y (pd.Series): The target variable.

        Returns:
            FeatureImportanceRegressionSelector: The fitted transformer instance.
        """
        logger.start_operation(
            f"Fitting FeatureImportanceRegressionSelector with top {self.k}% important features."
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        self.feature_importances_ = model.feature_importances_
        total_features = len(self.feature_importances_)
        num_features_to_select = int(np.ceil(total_features * self.k / 100))
        if num_features_to_select == 0:
            num_features_to_select = 1
        indices = np.argsort(self.feature_importances_)[-num_features_to_select:][::-1]
        self.selected_columns = X.columns[indices].tolist()

        logger.end_operation()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Selects the top k% of features most important according to the Random Forest model.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target variable (to append to the result).

        Returns:
            pd.DataFrame: The transformed data with only the selected top k% important features.
        """
        logger.start_operation(
            f"Transforming data by selecting {len(self.selected_columns)} most important features."
        )

        X_selected = X[self.selected_columns].copy()
        if y is not None:
            if isinstance(y, list):
                y = pd.Series(y)
            y_name = y.name if y.name is not None else "y"
            logger.debug(
                f"Appending target variable '{y_name}' to the transformed data."
            )
            X_selected[y_name] = y.values

        logger.end_operation()
        return X_selected

    def fit_transform(self, X: pd.DataFrame, y):
        """
        Fits and transforms the data by selecting the top k% most important features. Performs fit and transform in one step.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target variable.
        """

        logger.start_operation(
            f"Fitting and transforming data with top {self.k}% important features."
        )
        result = self.fit(X, y).transform(X, y)
        logger.end_operation()
        return result

    def to_tex(self) -> dict:
        """
        Returns a short description of the transformer in dictionary format.
        """
        return {
            "name": "FeatureImportanceRegressionSelector",
            "desc": f"Selects the top {self.k}% (rounded to whole number) of features most important according to Random Forest model for regression. Number of features that were selected: {len(self.selected_columns)}",
            "params": {"k": self.k},
        }

    def is_numerical(self) -> bool:
        return True
