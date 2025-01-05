from sklearn.svm import SVC

from ..utils.abstract import Classification
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelSVC(SVC, Classification):
    """
    Support Vector Classifier model.
    """

    PARAM_GRID = {
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [3, 4, 5],
        "gamma": ["scale", "auto"],
        "random_state": [42],
    }

    def __init__(self):
        """
        Initializes the Support Vector Classifier model.

        """
        super().__init__()
        logger.info("Support Vector Classifier model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "SVC",
            "desc": f"Support Vector Classifier model."
            f"Parameters: {self.get_params()}",
        }
