import warnings

from sklearn.svm import LinearSVR

from ..utils.abstract import Regressor
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelLinearSVR(LinearSVR, Regressor):
    """
    Linear SVR model with added description method (to_tex())
    and predefined PARAM_GRID that may be used in GridSearch.
    """

    PARAM_GRID = {
        "epsilon": [0.0, 0.1, 0.2, 0.5, 1.0],
        "C": [0.1, 1.0, 10.0, 100.0],
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "fit_intercept": [True, False],
    }

    def __init__(
        self, epsilon=0, C=1.0, loss="epsilon_insensitive", fit_intercept=True, **kwargs
    ):
        """
        Initializes Linear SVR model with specified parameters.
        Args:
            epsilon (float) : Epsilon parameter in the epsilon-insensitive loss function. Default:0.0
            C (float) : Regularization parameter.Default: 1.0.
            loss (str) : Loss function to be used. Default: 'epsilon_insensitive'.
            fit_intercept (bool) : whether to calculate intercept for this model. Default: True.
        """
        super().__init__(
            epsilon=epsilon, C=C, loss=loss, fit_intercept=fit_intercept, **kwargs
        )
        logger.start_operation("Initializing LinearSVR model.")
        warnings.filterwarnings("ignore")
        logger.end_operation()

    def to_tex(self) -> dict:
        """
        Returns a description of the model in a dictionary format.

        Returns:
            dict : a dictionary containing models name, description and hyperparameters.
        """
        return {
            "name": "Linear SVR",
            "desc": "Linear SVR models with hyperparameters.",
            "params": f"{self.get_params()}",
        }
