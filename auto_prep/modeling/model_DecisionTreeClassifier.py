from sklearn.tree import DecisionTreeClassifier

from ..utils.abstract import Classifier
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelDecisionTreeClassifier(DecisionTreeClassifier, Classifier):
    """
    This class extends the DecisionTreeClassifier and Classification classes to provide
    a decision tree classifier model with additional functionality.

    Attributes:
        PARAM_GRID (dict): A dictionary containing the parameter grid for hyperparameter tuning.

    Methods:
        to_tex() -> dict:
            Returns a short description in the form of a dictionary.
    """

    PARAM_GRID = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "random_state": [42],
    }

    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        **kwargs,
    ):
        """
        Initializes the Decision Tree Classifier model.

        Args:
            criterion (str): The function to measure the quality of a split. Default is "gini".
            splitter (str): The strategy used to choose the split at each node. Default is "best".
            max_depth (int or None): The maximum depth of the tree. Default is None.
            min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Default is 1.
            random_state (int): Controls the randomness of the estimator. Default is 42.
            **kwargs: Additional keyword arguments passed to the DecisionTreeClassifier.

        """
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "DecisionTreeClassifier",
            "desc": "Decision Tree Classifier model.",
            "params": f"{self.get_params()}",
        }
