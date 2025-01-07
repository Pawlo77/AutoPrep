from sklearn.neighbors import KNeighborsRegressor

from ..utils.abstract import Regressor
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelKNeighboursRegressor(KNeighborsRegressor, Regressor):
    """
    K Neighbours Regressor model.
    Attributes:
        PARAM_GRID (dict): Parameter grid for hyperparameter tuning.
    Methods:
        __init__(): Initializes the K Neighbours Regressor model.
        to_tex() -> dict: Returns a short description in the form of a dictionary.
    """

    PARAM_GRID = {
        "n_neighbors": [5, 10, 15],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [30, 40, 50],
        "p": [1, 2],
    }

    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        **kwargs,
    ):
        """
        Initializes the K Neighbours Regressor model.
        Args:
            n_neighbors (int, optional): Number of neighbors to use. Defaults to 5.
            weights (str, optional): Weight function used in prediction. Defaults to "uniform".
            algorithm (str, optional): Algorithm used to compute the nearest neighbors. Defaults to "auto".
            leaf_size (int, optional): Leaf size passed to BallTree or KDTree. Defaults to 30.
            p (int, optional): Power parameter for the Minkowski metric. Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            **kwargs,
        )
        logger.info("K Neighbours Regressor model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "KNeighboursRegressor",
            "desc": "K Neighbours Regressor model.",
            "params": f"{self.get_params()}",
        }
