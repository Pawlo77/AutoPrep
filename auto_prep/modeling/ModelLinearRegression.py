from sklearn.linear_model import LinearRegression

from ..utils.abstract import Regressor
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelLinearRegression(LinearRegression, Regressor):
    """
    Linear regression model with added description method (to_tex())
    and predefined PARAM_GRID that may be used in GridSearch.

    """

    PARAM_GRID = {
        "fit_intercept": [True, False],
    }

    def __init__(self, fit_intercept=True, **kwargs):
        """
        Initializes Linear Regression model with specified parameters.

        Args:
            fit_intercept (bool) : whether to calculate intercept for this model. Default: True.

        """
        super().__init__(fit_intercept=fit_intercept, **kwargs)
        logger.start_operation("Initializing Linear Regression model.")
        logger.end_operation()

    def to_tex(self) -> dict:
        """
        Returns a description of the model in a dictionary format.

        Returns:
            dict : a dictionary containing models name, description and hyperparameters.

        """
        return {
            "name": "Linear Regression",
            "desc": "Linear regression models with hyperparameters.",
            "params": f"{self.get_params()}",
        }
