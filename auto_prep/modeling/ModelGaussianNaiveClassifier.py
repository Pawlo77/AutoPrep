from sklearn.naive_bayes import GaussianNB

from ..utils.abstract import Classification
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelGaussianNaiveClassifier(GaussianNB, Classification):
    """
    Gaussian Naive Classifier model.
    """

    PARAM_GRID = {
        "priors": [None],
        "var_smoothing": [1e-9, 1e-7, 1e-5],
    }

    def __init__(self):
        """
        Initializes the Gaussian Naive Classifier model.

        """
        super().__init__()
        logger.info("Gaussian Naive Classifier model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "GaussianNaiveClassifier",
            "desc": f"Gaussian Naive Classifier model."
            f"Parameters: {self.get_params()}",
        }
