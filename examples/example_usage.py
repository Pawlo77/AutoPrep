import logging
from auto_prep.utils import config

config.update(
    perform_only_required_=False, raport_decimal_precision=2, log_level=logging.INFO
)

import openml
import numpy as np

from auto_prep.prep import AutoPrep

from sklearn.datasets import fetch_openml

# Load your dataset
# data = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto").frame
# data["survived"] = data["survived"].astype(np.uint8)


# data = openml.datasets.get_dataset(40945).get_data()[0]
# data["survived"] = data["survived"].astype(np.uint8)

data = openml.datasets.get_dataset(540)
X, _, _, _ = data.get_data(dataset_format="dataframe")

# data = openml.datasets.get_dataset(37)
# X, _, _, _ = data.get_data(dataset_format="dataframe")

# data = openml.datasets.get_dataset(338)
# X, _, _, _ = data.get_data(dataset_format="dataframe")

# Create and run pipeline
pipeline = AutoPrep()

if __name__ == "__main__":
    # pipeline.run(data, target_column="survived") # 40945
    # pipeline.run(X, target_column="GG_new") # 338
    pipeline.run(X, target_column="CL") # 540
