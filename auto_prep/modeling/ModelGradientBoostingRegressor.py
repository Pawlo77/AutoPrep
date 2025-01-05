from sklearn.ensemble import GradientBoostingRegressor

from ..utils.abstract import Regression
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelGradientBoostingRegressor(GradientBoostingRegressor, Regression):
    """
    Gradient Boosting Regressor model.
    """

    PARAM_GRID = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.1, 0.05, 0.02],
        "max_depth": [4, 6, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [1.0, 0.5],
        "random_state": [42],
    }

    def __init__(self):
        """
        Initializes the Gradient Boosting Regressor model.

        """
        super().__init__()
        logger.info("Gradient Boosting Regressor model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "GradientBoostingRegressor",
            "desc": f"Gradient Boosting Regressor model."
            f"Parameters: {self.get_params()}",
        }
