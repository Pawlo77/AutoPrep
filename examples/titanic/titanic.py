import logging
from sklearn.datasets import fetch_openml
from auto_prep.utils import config

config.set(
    perform_only_required_=True, raport_decimal_precision=2, log_level=logging.INFO
)

from auto_prep.prep import AutoPrep

# Load your dataset
data = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto").frame
data["survived"] = data["survived"]

pipeline = AutoPrep()

if __name__ == "__main__":
    pipeline.run(data, target_column="survived")
