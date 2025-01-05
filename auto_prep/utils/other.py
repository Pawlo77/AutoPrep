import os

import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

from .config import config
from .logging_config import setup_logger

logger = setup_logger(__name__)


def save_chart(name: str, *args, **kwargs) -> str:
    """
    Saves chart to directory specified in config.

    Args:
        name (str) - Chart name with extension.
    Returs:
        path (str) - Path where chart has been saved.
    """
    path = os.path.join(config.charts_dir, name)
    logger.debug(f"Saving to {path}...")
    plt.savefig(os.path.join(config.charts_dir, name), *args, **kwargs)
    plt.close()
    return path


def save_model(name: str, model: BaseEstimator, *args, **kwargs) -> str:
    """
    Saves model to directory specified in config.

    Args:
        name (str) - Chart name with extension.
        model (BaseEstimator) - Model to be saved.

    Returs:
        path (str) - Path where model has been saved.
    """
    path = os.path.join(config.models_dir, name)
    logger.debug(f"Saving to {path}...")
    joblib.dump(model, path)
    return path
