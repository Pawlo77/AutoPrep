from sklearn.neighbors import KNeighborsRegressor

from ..utils.abstract import Regression
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelKNeighboursRegressor(KNeighborsRegressor, Regression):
    """
    K Neighbours Regressor model.
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
        Initializes the K Neighbours Regressor model.

        """
        super().__init__()
        logger.info("K Neighbours Regressor model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "KNeighboursRegressor",
            "desc": f"K Neighbours Regressor model." f"Parameters: {self.get_params()}",
        }
