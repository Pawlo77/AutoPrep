import json
import os
from typing import Any, Union

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator

from .config import config
from .logging_config import setup_logger

logger = setup_logger(__name__)


def get_scoring(task: str, y_train: pd.Series) -> Union[callable, str]:
    """
    Retrieve proper scoring function from config.

    Args:
        y_train (pd.Series): Training target dataset.
        task (str): regiression / classification
    """
    if task == "classification":
        if len(y_train.unique()) > 2:
            return config.classification_pipeline_scoring_func_multi
        else:
            return config.classification_pipeline_scoring_func
    else:
        return config.regression_pipeline_scoring_func


def save_json(name: str, obj: Any) -> str:
    """
    Saves json-like object to directory specified in config.

    Args:
        name (str) - Chart name with extension.
    Returs:
        path (str) - Path where chart has been saved.
    """
    path = os.path.join(config.raport_path, name)
    logger.debug(f"Saving to {path}...")
    with open(path, "w") as file:
        json.dump(obj, file, indent=2)


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
    path = os.path.join(config.pipelines_dir, name)
    logger.debug(f"Saving to {path}...")
    joblib.dump(model, path)
    return path
