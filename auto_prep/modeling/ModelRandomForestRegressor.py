from sklearn.ensemble import RandomForestRegressor

from ..utils.abstract import Regressor
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelRandomForestRegressor(RandomForestRegressor, Regressor):
    """
    Random Forest Regressor model.
    Attributes:
        PARAM_GRID (dict): Parameter grid for hyperparameter tuning.
    Methods:
        __init__(): Initializes the Random Forest Regressor model.
        to_tex() -> dict: Returns a short description in the form of a dictionary.
    """

    PARAM_GRID = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "random_state": [42],
    }

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1,
        bootstrap=True,
        random_state=42,
        **kwargs,
    ):
        """
        Initializes the Random Forest Regressor model.
        Args:
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.
            criterion (str, optional): Function to measure the quality of a split. Defaults to "mse".
            max_depth (int, optional): Maximum depth of the tree. Defaults to None.
            min_samples_split (int, optional): Minimum number of samples required to split an internal node. Defaults to 2.
            min_samples_leaf (int, optional): Minimum number of samples required to be at a leaf node. Defaults to 1.
            max_features (str, optional): Number of features to consider when looking for the best split. Defaults to "auto".
            bootstrap (bool, optional): Whether bootstrap samples are used when building trees. Defaults to True.
            random_state (int, optional): Controls both the randomness of the bootstrapping of the samples used when building trees. Defaults to 42.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            **kwargs,
        )
        logger.info("Random Forest Regressor model initialized.")

    def to_tex(self) -> dict:
        """
        Returns a short description in form of dictionary.

        Returns:
            dict: A dictionary containing the name and description of the model.
        """
        return {
            "name": "RandomForestRegressor",
            "desc": "Random Forest Regressor model.",
            "params": f"{self.get_params()}",
        }
