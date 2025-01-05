from sklearn.ensemble import RandomForestRegressor

from ..utils.abstract import Regression
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelRandomForestRegressor(RandomForestRegressor, Regression):
    """
    Random Forest Regressor model.
    """

    PARAM_GRID = {
        "n_estimators": [100, 200, 300],
        "criterion": ["mse", "mae"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
        "random_state": [42],
    }

    def __init__(self):
        """
        Initializes the Random Forest Regressor model.

        """
        super().__init__()
        logger.info("Random Forest Regressor model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "RandomForestRegressor",
            "desc": f"Random Forest Regressor model."
            f"Parameters: {self.get_params()}",
        }
