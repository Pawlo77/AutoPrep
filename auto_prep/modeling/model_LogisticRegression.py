from sklearn.linear_model import LogisticRegression

from ..utils.abstract import Classifier
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelLogisticRegression(LogisticRegression, Classifier):
    """
    Logistic regression model with added description method (to_tex())
    and predefined PARAM_GRID that may be used in GridSearch.

    """

    PARAM_GRID = [
        {"penalty": ["l1"], "C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "saga"]},
        {
            "penalty": ["l2"],
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear", "saga", "newton-cg"],
        },
        {
            "penalty": ["elasticnet"],
            "C": [0.01, 0.1, 1, 10],
            "solver": ["saga"],
            "l1_ratio": [0.5, 0.7],
        },
    ]

    def __init__(self, penalty="l2", C=1.0, solver="lbfgs", l1_ratio=None, **kwargs):
        """
        Initializes Logistic Regression model with specified hyperparameters.
        Args:
            penalty (str) : Specify the norm of the penalty. Default: 'l2'.
            C (float) : Inverse of regularization strength. Default:1.0.
            solver (str) : Algorithm to use in the optimization problem. Default: 'lbfgs'.
            l1_ratio (float) : The Elastic-Net mixing parameter. Default:None.

        """
        super().__init__(
            penalty=penalty, C=C, solver=solver, l1_ratio=l1_ratio, **kwargs
        )

    def to_tex(self) -> dict:
        """
        Returns a description of the model in dictionary format.

        Returns:
            dict : a dictionary containing models name, description and hyperparameters.

        """
        return {
            "name": "Logistic Regression",
            "desc": "Logistic regression models with hyperparameters.",
            "params": f"{self.get_params()}",
        }
