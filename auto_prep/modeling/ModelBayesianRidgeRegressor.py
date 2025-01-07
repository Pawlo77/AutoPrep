from sklearn.linear_model import BayesianRidge

from ..utils.abstract import Regressor
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelBayesianRidgeRegressor(BayesianRidge, Regressor):
    """
    This class implements a Bayesian Ridge Regressor model, which is a linear
    regression model with Bayesian regularization.

    Attributes:
        PARAM_GRID (dict): A dictionary containing the parameter grid for
            hyperparameter tuning.
    Methods:
        to_tex() -> dict:
            Returns a short description in the form of a dictionary.

    """

    PARAM_GRID = {
        "max_iter": [300, 400, 500],
        "tol": [1e-3, 1e-4, 1e-5],
        "alpha_1": [1e-6, 1e-7, 1e-8],
        "alpha_2": [1e-6, 1e-7, 1e-8],
        "lambda_1": [1e-6, 1e-7, 1e-8],
        "lambda_2": [1e-6, 1e-7, 1e-8],
    }

    def __init__(
        self,
        max_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        **kwargs,
    ):
        """
        Initializes the Bayesian Ridge Regressor model.

        Args:
        max_iter (int, optional): Maximum number of iterations. Default is 300.
        tol (float, optional): Tolerance for the stopping criterion. Default is 1e-3.
        alpha_1 (float, optional): Hyperparameter for the shape parameter of the Gamma
            distribution prior over the alpha parameter. Default is 1e-6.
        alpha_2 (float, optional): Hyperparameter for the inverse scale parameter of the
            Gamma distribution prior over the alpha parameter. Default is 1e-6.
        lambda_1 (float, optional): Hyperparameter for the shape parameter of the Gamma
            distribution prior over the lambda parameter. Default is 1e-6.
        lambda_2 (float, optional): Hyperparameter for the inverse scale parameter of the
            Gamma distribution prior over the lambda parameter. Default is 1e-6.
        **kwargs: Additional keyword arguments passed to the parent class.

        """
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            **kwargs,
        )
        logger.info("Bayesian Ridge Regressor model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "BayesianRidgeRegressor",
            "desc": "Bayesian Ridge Regressor model.",
            "params": f"{self.get_params()}",
        }
