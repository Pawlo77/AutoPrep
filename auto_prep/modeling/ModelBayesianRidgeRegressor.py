from sklearn.linear_model import BayesianRidge

from ..utils.abstract import Regression
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelBayesianRidgeRegressor(BayesianRidge, Regression):
    """
    Bayesian Ridge Regressor model.
    """

    PARAM_GRID = {
        "max_iter": [300, 400, 500],
        "tol": [1e-3, 1e-4, 1e-5],
        "alpha_1": [1e-6, 1e-7, 1e-8],
        "alpha_2": [1e-6, 1e-7, 1e-8],
        "lambda_1": [1e-6, 1e-7, 1e-8],
        "lambda_2": [1e-6, 1e-7, 1e-8],
    }

    def __init__(self):
        """
        Initializes the Bayesian Ridge Regressor model.

        """
        super().__init__()
        logger.info("Bayesian Ridge Regressor model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "BayesianRidgeRegressor",
            "desc": f"Bayesian Ridge Regressor model."
            f"Parameters: {self.get_params()}",
        }
