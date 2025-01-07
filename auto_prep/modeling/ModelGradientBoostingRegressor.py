from sklearn.ensemble import GradientBoostingRegressor

from ..utils.abstract import Regressor
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelGradientBoostingRegressor(GradientBoostingRegressor, Regressor):
    """
    This class implements a Gradient Boosting Regressor model with a predefined parameter grid
    for hyperparameter tuning.
    Attributes:
        PARAM_GRID (dict): A dictionary containing the parameter grid for hyperparameter tuning.
    Methods:
        __init__(): Initializes the Gradient Boosting Regressor model.
        to_tex() -> dict: Returns a short description in the form of a dictionary.
        This method initializes the Gradient Boosting Regressor model and logs the initialization.

    """

    PARAM_GRID = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.1, 0.05, 0.02],
        "max_depth": [4, 6, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [1.0, 0.5],
        "random_state": [42],
    }

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42,
        **kwargs,
    ):
        """
        Initializes the Gradient Boosting Regressor model.

        Args:
            n_estimators (int): The number of boosting stages to be run.
            learning_rate (float): The learning rate shrinks the contribution of each tree.
            max_depth (int): The maximum depth of the individual regression estimators.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            subsample (float): The fraction of samples to be used for fitting the individual base learners.
            random_state (int): The seed used by the random number generator.
            **kwargs: Additional keyword arguments passed to the GradientBoostingRegressor class.
        """
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            **kwargs,
        )
        logger.info("Gradient Boosting Regressor model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "GradientBoostingRegressor",
            "desc": "Gradient Boosting Regressor model.",
            "params": f"{self.get_params()}",
        }
