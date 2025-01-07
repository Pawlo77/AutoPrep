from xgboost import XGBClassifier

from ..utils.abstract import Classifier
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelXGBoost(XGBClassifier, Classifier):
    """
    XGBoost classification model with added description method (to_tex())
    and predefined PARAM_GRID that may be used in GridSearch.
    """

    PARAM_GRID = {
        "max_depth": [3, 6, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.5, 0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "objective": ["binary:logistic", "multi:softprob", "reg:squarederror"],
    }

    def __init__(
        self,
        max_depth=6,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=0.8,
        objective="binary:logistic",
        **kwargs,
    ):
        """
        Initializes XGBoost model with specified hyperparameters.

        Args:
            max_depth (int) : Maximum depth of a tree. Default: 6.
            learning_rate (float) : Step size shrinkage used in update to prevent overfitting. Default: 0.3.
            subsample (float) : Subsample ratio of the training instances. Default: 1.0.
            colsample_bytree (float) : Subsample ratio of columns when constructing each tree. Default: 0.8.
            objective (str) : Loss function to be used. Default: 'binary:logistic'
        """
        super().__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            **kwargs,
        )
        logger.start_operation("Initializing XGBoost model")
        logger.end_operation()

    def to_tex(self) -> dict:
        """
        Returns a description of the model in a dictionary format.

        Returns:
            dict : a dictionary containing models name, description and hyperparameters.

        """
        params = {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
        return {
            "name": "XGBoost",
            "desc": "XGBoost model with hyperparameters.",
            "params": params,
        }
