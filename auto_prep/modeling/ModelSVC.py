from sklearn.svm import SVC

from ..utils.abstract import Classifier
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelSVC(SVC, Classifier):
    """
    Support Vector Classifier model.
    Attributes:
        PARAM_GRID (dict): Parameter grid for hyperparameter tuning.
    Methods:
        __init__(): Initializes the Support Vector Classifier model.
        to_tex() -> dict: Returns a short description in the form of a dictionary.
    """

    PARAM_GRID = {
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [3, 4, 5],
        "gamma": ["scale", "auto"],
        "random_state": [42],
    }

    def __init__(
        self, C=1.0, kernel="rbf", degree=3, gamma="scale", random_state=42, **kwargs
    ):
        """
        Initializes the Support Vector Classifier model.
        Args:
            C (float, optional): Regularization parameter. Defaults to 1.0.
            kernel (str, optional): Specifies the kernel type to be used in the algorithm. Defaults to "rbf".
            degree (int, optional): Degree of the polynomial kernel function. Defaults to 3.
            gamma (str, optional): Kernel coefficient. Defaults to "scale".
            random_state (int, optional): Seed for random number generator. Defaults to 42.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            random_state=random_state,
            **kwargs,
        )
        logger.info("Support Vector Classifier model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "SVC",
            "desc": "Support Vector Classifier model.",
            "params": f"{self.get_params()}",
        }
