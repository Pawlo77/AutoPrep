from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Numerical(ABC):
    """Abstract interface to indicate numerical step"""

    pass


class Categorical(ABC):
    """Abstract interface to indicate categorical step"""

    pass


class Step(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract class to be overwritten for implementing custom
    preprocessing steps. If step is parametrizable, it should have
    defined "param_grid" of all possible values for each parameter.
    """

    @abstractmethod
    def to_tex(self) -> dict:
        pass

    @abstractmethod
    def is_numerical(self) -> bool:
        """
        If this step is for numerical data or not.

        Returns:
            bool: True if the step is for numerical data, False otherwise.
        """
        pass


class RequiredStep(Step):
    """
    Required step that will be always considered in preprocessing.
    """

    pass


class NonRequiredStep(Step):
    """
    Non required step that will be only considered for preprocessing
    if class method is_applicable returns True.
    """

    @staticmethod
    @abstractmethod
    def is_applicable(dt: pd.Series) -> bool:
        """
        Args:
            dt (pd.Series) - column that is considered to be preprocessed
                by that transformer.

        Returns:
            bool - True if it is possible to use this transofmation
                on passed data.
        """
        pass
