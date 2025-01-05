from sklearn.tree import DecisionTreeClassifier

from ..utils.abstract import Classification
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelDecisionTreeClassifier(DecisionTreeClassifier, Classification):
    """
    Decision Tree Classifier model.
    """

    PARAM_GRID = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "random_state": [42],
    }

    def __init__(self):
        """
        Initializes the Decision Tree Classifier model.

        """
        super().__init__()
        logger.info("Decision Tree Classifier model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "DecisionTreeClassifier",
            "desc": f"Decision Tree Classifier model."
            f"Parameters: {self.get_params()}",
        }
