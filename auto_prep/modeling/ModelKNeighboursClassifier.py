from sklearn.neighbors import KNeighborsClassifier

from ..utils.abstract import Classification
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelKNeighboursClassifier(KNeighborsClassifier, Classification):
    """
    K Neighbours Classifier model.
    """

    PARAM_GRID = {
        "n_neighbors": [5, 10, 15],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [30, 40, 50],
        "p": [1, 2],
    }

    def __init__(self):
        """
        Initializes the K Neighbours Classifier model.

        """
        super().__init__()
        logger.info("K Neighbours Classifier model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "KNeighboursClassifier",
            "desc": f"K Neighbours Classifier model."
            f"Parameters: {self.get_params()}",
        }
