import logging
from auto_prep.utils import config

config.update(
    perform_only_required_=False, raport_decimal_precision=2, log_level=logging.INFO
)

import numpy as np

from auto_prep.prep import AutoPrep
from sklearn.datasets import fetch_openml

# Load your dataset
data = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto").frame
data["survived"] = data["survived"].astype(np.uint8)

# Create and run pipeline
pipeline = AutoPrep()

if __name__ == "__main__":
    pipeline.run(data, target_column="survived")
