from sklearn.naive_bayes import GaussianNB

from ..utils.abstract import Classifier
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelGaussianNaiveClassifier(GaussianNB, Classifier):
    """
    This class implements a Gaussian Naive Bayes classifier, which is a probabilistic
    classifier based on applying Bayes' theorem with strong (naive) independence
    assumptions between the features.
    Attributes:
        PARAM_GRID (dict): A dictionary containing the parameter grid for hyperparameter
            tuning. It includes:
            - "priors": List of prior probabilities of the classes. Default is [None].
            - "var_smoothing": List of float values for the portion of the largest
              variance of all features that is added to variances for calculation stability.
    Methods:
        __init__():
        to_tex() -> dict:
            Returns a short description in the form of a dictionary.
    """

    PARAM_GRID = {
        "priors": [None],
        "var_smoothing": [1e-9, 1e-7, 1e-5],
    }

    def __init__(self, priors=None, var_smoothing=1e-9, **kwargs):
        """
        Initializes the Gaussian Naive Classifier model.

        Args:
            priors (list): List of prior probabilities of the classes. Default is None.
            var_smoothing (float): Portion of the largest variance of all features that is
                added to variances for calculation stability. Default is 1e-9.
            **kwargs: Additional keyword arguments passed to the GaussianNB.
        """
        super().__init__(priors=priors, var_smoothing=var_smoothing, **kwargs)
        logger.info("Gaussian Naive Classifier model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "GaussianNaiveClassifier",
            "desc": "Gaussian Naive Classifier model.",
            "params": f"{self.get_params()}",
        }
