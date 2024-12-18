import numpy as np
from auto_prep.pipeline import AutoMLPipeline
from sklearn.datasets import fetch_openml

# Load your dataset
data = fetch_openml(name="titanic", version=1, as_frame=True).frame
data["survived"] = data["survived"].astype(np.uint8)

# Create and run pipeline
pipeline = AutoMLPipeline()
pipeline.run(data, target_column="survived")
