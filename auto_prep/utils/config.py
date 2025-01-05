import logging
import os
from typing import List

import numpy as np
from pylatex import NoEscape

# ANSI color codes
COLORS: dict = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
    "RESET": "\033[0m",  # Reset color
}

LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = logging.INFO

DEFAULT_TEX_GEOMETRY: dict = {
    "margin": "0.5in",
    "headheight": "10pt",
    "footskip": "0.2in",
    "tmargin": "0.5in",
    "bmargin": "0.5in",
}

DEFAULT_ABSTRACT: str = NoEscape(
    r"""
    \begin{abstract}
    This raport has been generated with AutoPrep.
    \end{abstract}
    """
)


class GlobalConfig:
    """Global config class."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalConfig, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(
        self,
        raport_name: str = "raport",
        raport_title: str = "ML Raport",
        raport_author: str = "AutoPrep",
        raport_abstract: str = DEFAULT_ABSTRACT,
        root_dir: str = "raport",
        return_tex_: bool = True,
        logger_colors_map: dict = COLORS,
        log_format: str = LOG_FORMAT,
        log_date_format: str = LOG_DATE_FORMAT,
        log_level: str = LOG_LEVEL,
        log_dir: str = None,
        max_log_file_size_in_mb: int = 5,
        tex_geomatry: dict = DEFAULT_TEX_GEOMETRY,
        train_size: float = 0.8,
        test_size: float = 0.1,
        valid_size: float = 0.1,
        random_state: int = 42,
        max_datasets_after_preprocessing: int = 3,
        perform_only_required_: bool = False,
        raport_decimal_precision: int = 4,
        raport_chart_color_pallete: List[str] = ["#FF204E"],
        correlation_threshold: float = 0.8,
        correlation_percent: float = 0.5,
        n_bins: int = 4,
        chart_settings: dict = None,
    ):
        """
        Args:
            raport_name (str) - Raport name. Defaults to "raport.pdf".
            raport_title (str) - Raport title. Defaults to "ML Raport".
            raport_title (str) - Raport author. Defaults to "AutoPrep".
            raport_abstract (str) - Raport abstract section. Can be set to "".
                Defaults to :obj:`DEFAULT_ABSTRACT`.
            root_dir (str) - Root directory. Here raport will be
                stored and all cache. Defaults to "raport".
            return_tex_ (bool) - If true it will create .tex file
                alongsite the pdf. Defaults to True.
            logger_colors_map (dict) - Color map for the loggers.
                Defaults to :obj:`COLORS`.
            log_format (str) - Log format for logging liblary.
                Defaults to :obj:`LOG_FORMAT`.
            log_date_format (str) - Log date format for logging liblary.
                Defaults to :obj:`LOG_DATE_FORMAT`.
            log_level (str) - Log level for logging liblary.
                Defaults to :obj:`LOG_LEVEL`.
            log_dir (str) - Log directory for storing the logs.
                If None provided, will default to "logs" in root dir.
                -1 means no logging to file.
            max_log_file_size_in_mb (int) - Maximum file size in mb for
                each logger. Defaults to 5.
            tex_geomatry (dict) - Geometry for pylatex.
                Defaults to :obj:`DEFAULT_TEX_GEOMETRY`.
            train_size (float) - % of traing set size. Defaults to 0.8.
            test_size (float) - % of traing set size. Defaults to 0.1.
            valid_size (float) - % of traing set size. Defaults to 0.1.
            random_state (int) - Random state for sklearn.
            max_datasets_after_preprocessing (int) - Maximum number of datasets that will be left
                after preprocessing steps. On them further models will be trained. Strongly
                affects performance.
            perform_only_required_ (bool) - weather or not to perform only required steps.
                Affects entire process.
            raport_decimal_precision (int) - Decimal precision for all float in raport.
                Will use standard python rounding.

            raport_chart_color_pallete (List[str]) - Color palette for basic eda charts.
            correlation_threshold (float) - threshold used for detecting highly correlated features.Default 0.8.
            correlation_percent (float) - % of selected features based on their correlation with the target. Default 0.5.
            n_bins (int) - number of bins to create while binning numerical features.

            chart_settings (dict): Settings for customizing chart appearance.
                Defaults to None, which initializes default settings.

        """
        assert (
            isinstance(raport_name, str) and raport_name != ""
        ), "raport_name should not be empty"
        self.raport_name = raport_name
        self.raport_title = raport_title
        self.raport_author = raport_author
        self.raport_abstract = raport_abstract
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir
        self.raport_path = os.path.abspath(os.path.join(root_dir, raport_name))
        self.charts_dir = os.path.join(self.raport_path, "charts")
        os.makedirs(self.charts_dir, exist_ok=True)
        self.return_tex_ = return_tex_

        self.logger_colors_map = logger_colors_map
        self.log_format = log_format
        self.log_date_format = log_date_format
        self.log_level = log_level

        assert (
            int(max_log_file_size_in_mb) == max_log_file_size_in_mb
            and max_log_file_size_in_mb >= 1
        ), f"Wrong value for max_log_file_size_in_mb: {max_log_file_size_in_mb}. "
        "Should be int > 1."
        self.max_log_file_size_in_mb = max_log_file_size_in_mb

        if log_dir is None:
            log_dir = os.path.join(root_dir, "logs")
        if log_dir != -1:
            os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        self.tex_geomatry = tex_geomatry

        self.train_size = train_size
        self.test_size = test_size
        self.valid_size = valid_size

        self.random_state = random_state
        np.random.seed(random_state)

        self.raport_chart_color_pallete = raport_chart_color_pallete
        self.chart_settings = chart_settings or {
            "theme": "whitegrid",
            "title_fontsize": 18,
            "title_fontweight": "bold",
            "xlabel_fontsize": 12,
            "ylabel_fontsize": 12,
            "tick_label_rotation": 45,
            "palette": "pastel",
            "plot_width": 15,
            "plot_height_per_row": 4,
            "heatmap_cmap": "coolwarm",
            "heatmap_fmt": ".2f",
        }

        assert (
            max_datasets_after_preprocessing > 0
        ), "Values smaller than 1 are forbidden."
        self.max_datasets_after_preprocessing = max_datasets_after_preprocessing
        self.perform_only_required_ = perform_only_required_

        self.raport_decimal_precision = raport_decimal_precision

        self.root_project_dir = os.path.abspath("./..")

        assert 0 <= correlation_threshold <= 1, (
            f"Invalid value for correlation_threshold: {correlation_threshold}. "
            "It must be a float between 0 and 1."
        )
        self.correlation_threshold = correlation_threshold

        assert 0 <= correlation_percent <= 1, (
            f"Invalid value for correlation_selector_percent: {correlation_percent}. "
            "It must be a float between 0 and 1."
        )
        self.correlation_percent = correlation_percent

        assert (
            int(n_bins) == n_bins and n_bins >= 1
        ), f"Wrong value for n_bins: {n_bins}. "
        "Should be int >= 1."
        self.n_bins = n_bins

    def update(self, **kwargs):
        """Updates config's data with kwargs."""
        for key, value in kwargs.items():
            setattr(self, key, value)


config = GlobalConfig()
