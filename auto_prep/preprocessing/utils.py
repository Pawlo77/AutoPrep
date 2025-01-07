import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, column_or_1d

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class TolerantLabelEncoder(LabelEncoder):
    def __init__(
        self,
        ignore_unknown=True,
        unknown_original_value="unknown",
        unknown_encoded_value=-1,
    ):
        self.ignore_unknown = ignore_unknown
        self.unknown_original_value = unknown_original_value
        self.unknown_encoded_value = unknown_encoded_value

    def transform(self, y, column):
        check_is_fitted(self, "classes_")
        y = column_or_1d(y, warn=True)

        indices = np.isin(y, self.classes_)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError(
                f"{column} contains new labels: {np.setdiff1d(y, self.classes_)}"
            )
        elif not np.all(indices):
            logger.warning(
                f"{column} contains new labels: {len(np.setdiff1d(y, self.classes_))}"
            )

        y_transformed = np.searchsorted(self.classes_, y)
        y_transformed[~indices] = self.unknown_encoded_value
        return y_transformed

    def inverse_transform(self, y):
        check_is_fitted(self, "classes_")

        labels = np.arange(len(self.classes_))
        indices = np.isin(y, labels)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError(
                "y contains new labels: %s" % str(np.setdiff1d(y, self.classes_))
            )

        y_transformed = np.asarray(self.classes_[y], dtype=object)
        y_transformed[~indices] = self.unknown_original_value
        return y_transformed
