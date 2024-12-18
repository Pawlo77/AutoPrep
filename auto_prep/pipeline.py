import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pylatex import NoEscape
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing.cleaner import DataCleaner
from .preprocessing.encoder import AdvancedEncoder
from .reporting.latex_generator import ReportGenerator
from .utils.logging_config import setup_logger
from .utils.system import get_system_info
from .visualization.eda import EDAVisualizer

logger = setup_logger(__name__)


class AutoMLPipeline:
    """Main pipeline orchestrating the entire preprocessing process.

    This class handles the complete workflow from data preprocessing to
    report generation.

    Attributes:
        output_dir (str): Directory where reports and figures will be saved.
        visualizer (EDAVisualizer): Component for generating visualizations.
        report_generator (ReportGenerator): Component for generating LaTeX reports.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        """Initialize the pipeline with output directory.

        Args:
            output_dir (str, optional): Directory for saving outputs.
                Defaults to "reports".
        """
        logger.info("Initializing AutoMLPipeline with output_dir: %s", output_dir)
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(os.path.join(".", output_dir))
        self._output_dir = output_dir
        self._figure_dir = os.path.join(self._output_dir, "figures")

        try:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(self._figure_dir, exist_ok=True)
            logger.debug("Created output directories")
        except Exception as e:
            logger.error("Failed to create output directories: %s", str(e))
            raise

        self.visualizer = EDAVisualizer(self._figure_dir)
        self.report_generator = ReportGenerator()
        logger.debug("Initialized components")

    def run(self, data: pd.DataFrame, target_column: str) -> None:
        """Run the complete pipeline on the provided dataset.

        Args:
            data (pd.DataFrame): Input dataset to process.
            target_column (str): Name of the target variable column.
        """
        logger.info("Starting pipeline run with target column: %s", target_column)
        logger.debug("Input data shape: %s", data.shape)

        data = data.rename(columns=lambda x: x.replace(".", "__"))

        try:
            # Split features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            logger.debug("Split features and target")

            # Generate summaries and EDA
            self._generate_dataset_summary(data, target_column)
            self._generate_eda(X, y)
            logger.info("Generated dataset summary and EDA")

            # Create and fit preprocessing pipeline
            preprocess_pipeline = Pipeline(
                [("cleaner", DataCleaner()), ("encoder", AdvancedEncoder())]
            )
            logger.debug("Created preprocessing pipeline")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fit and transform the preprocessing pipeline
            X_train_transformed = preprocess_pipeline.fit_transform(X_train)

            # Create and fit classifier
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_train_transformed, y_train)

            # Create full pipeline for cross-validation
            full_pipeline = Pipeline(
                [
                    ("preprocessing", preprocess_pipeline),
                    (
                        "classifier",
                        RandomForestClassifier(n_estimators=100, random_state=42),
                    ),
                ]
            )

            # Calculate cross-validation scores
            cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5)

            # Transform test data and get predictions
            X_test_transformed = preprocess_pipeline.transform(X_test)
            y_pred = classifier.predict(X_test_transformed)

            # Generate model performance visualizations
            self._generate_model_performance(y_test, y_pred)

            # Generate feature importance plot if available
            if hasattr(classifier, "feature_importances_"):
                self._generate_feature_importance(
                    classifier,
                    X_train_transformed.columns,
                    self._get_fig_path("feature_importance.png"),
                )

            # Generate report
            self._generate_report(
                data,
                target_column,
                classifier,
                X_test_transformed,
                y_test,
                y_pred,
                cv_scores,
            )

        except Exception as e:
            logger.error("Pipeline run failed: %s", str(e))
            raise

    def _generate_dataset_summary(self, data: pd.DataFrame, target_column: str) -> None:
        """Generate summary statistics for the dataset.

        Args:
            data (pd.DataFrame): Full dataset.
            target_column (str): Name of target variable.
        """
        logger.debug("Generating dataset summary")
        try:
            # Get feature types
            features = data.drop(columns=[target_column])

            numeric_features = features.select_dtypes(
                include=["int64", "float64"]
            ).columns
            categorical_features = features.select_dtypes(
                include=["object", "category"]
            ).columns

            # Basic statistics
            self.dataset_summary = {
                "n_samples": len(data),
                "n_features": len(features.columns),
                "n_numeric": len(numeric_features),
                "n_categorical": len(categorical_features),
                "numeric_features": list(numeric_features),
                "categorical_features": list(categorical_features),
                "target_distribution": data[target_column].value_counts().to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
            }
            logger.debug(
                f"Found {len(numeric_features)} numeric and "
                "{len(categorical_features)} categorical features"
            )
        except Exception as e:
            logger.error(f"Failed to generate dataset summary: {str(e)}")
            raise

    def _generate_eda(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Generate exploratory data analysis visualizations.

        Args:
            X (pd.DataFrame): Feature dataset.
            y (pd.Series): Target variable.
        """
        logger.info("Generating exploratory data analysis")
        try:
            # Generate target distribution plot
            self.visualizer.generate_target_distribution(y)
            logger.debug("Generated target distribution plot")

            # Generate numeric feature plots
            if self.dataset_summary["numeric_features"]:
                self.visualizer.generate_distribution_plots(
                    X, self.dataset_summary["numeric_features"]
                )
                logger.debug("Generated numeric feature distribution plots")

            # Generate categorical feature plots
            if self.dataset_summary["categorical_features"]:
                self.visualizer.generate_categorical_plots(
                    X, self.dataset_summary["categorical_features"]
                )
                logger.debug("Generated categorical feature plots")

            # Generate correlation matrix
            self.visualizer.generate_correlation_matrix(X)
            logger.debug("Generated correlation matrix")

            # Generate missing values plot
            if any(self.dataset_summary["missing_values"].values()):
                self.visualizer.generate_missing_values_plot(X)
                logger.debug("Generated missing values plot")

            # Generate target correlations
            target_corr_path = self.visualizer.generate_target_correlations(X, y)
            if target_corr_path:
                self.target_correlations_available = True
                logger.debug("Generated target correlations plot")

        except Exception as e:
            logger.error(f"Failed to generate EDA visualizations: {str(e)}")
            raise

    def _generate_model_performance(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> None:
        """Generate model performance visualizations.

        Args:
            y_true (pd.Series): True target values.
            y_pred (np.ndarray): Predicted target values.
        """
        # Generate confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(y_true, y_pred, normalize="true"),
            annot=True,
            fmt=".2f",
            cmap="Blues",
        )
        plt.title("Normalized Confusion Matrix")
        plt.savefig(self._get_fig_path("confusion_matrix.png"))
        plt.close()

    def _generate_feature_importance(
        self,
        classifier: RandomForestClassifier,
        feature_names: pd.Index,
        output_path: str,
    ) -> None:
        """Generate feature importance plot.

        Args:
            classifier (RandomForestClassifier): Trained classifier.
            feature_names (pd.Index): Feature names.
            output_path (str): Path to save the plot.
        """
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(
            range(len(importances)), feature_names[indices], rotation=45, ha="right"
        )
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _generate_report(
        self,
        data: pd.DataFrame,  # noqa: F841
        target_column: str,  # noqa: F841
        classifier: RandomForestClassifier,
        X_test: pd.DataFrame,  # noqa: F841
        y_test: pd.Series,
        y_pred: np.ndarray,
        cv_scores: np.ndarray,
    ) -> None:
        """
        Generates the final LaTeX report with analysis results.

        Args:
            data (pd.DataFrame): The input dataset.
            target_column (str): The name of the target column in the dataset.
            classifier (RandomForestClassifier): The trained classifier model.
            X_test (pd.DataFrame): The test dataset features.
            y_test (pd.Series): The true target values for the test dataset.
            y_pred (np.ndarray): The predicted target values.
            cv_scores (np.ndarray): The cross-validation scores.
        """
        logger.info("Generating analysis report")
        try:
            self.report_generator.add_header()

            # Overview section
            overview_section = self.report_generator.add_section(
                "Overview"
            )  # noqa: F841
            system_subsection = self.report_generator.add_subsection(
                "System"
            )  # noqa: F841
            self._dict_to_latex_table(
                get_system_info(),
                header=None,
            )
            dataset_subsection = self.report_generator.add_subsection(
                "Dataset"
            )  # noqa: F841
            self._dict_to_latex_table(
                {
                    "Number of samples": f"{self.dataset_summary['n_samples']}",
                    "Number of features": f"{self.dataset_summary['n_features']}",
                    "Numeric features": f"{self.dataset_summary['n_numeric']}",
                    "Categorical features": f"{self.dataset_summary['n_categorical']}",
                },
                header=None,
            )

            # Add Target Distribution subsection
            target_subsection = self.report_generator.add_subsection(
                "Target Distribution"
            )  # noqa: F841
            self._dict_to_latex_table(
                self.dataset_summary["target_distribution"], "Target Distribution"
            )
            # Add target distribution plot
            self.report_generator.add_figure(
                self._get_fig_path("target_distribution.png"),
                "Target Variable Distribution",
            )

            # Add Missing Values subsection
            missing_subsection = self.report_generator.add_subsection("Missing Values")
            if not any(self.dataset_summary["missing_values"].values()):
                missing_subsection.append("No missing values found in the dataset.")
            else:
                self._dict_to_latex_table(
                    self.dataset_summary["missing_values"], "Missing Values"
                )
                self.report_generator.add_figure(
                    self._get_fig_path("missing_values.png"), "Missing Values Analysis"
                )

            # Add EDA section
            eda_section = self.report_generator.add_section(
                "Exploratory Data Analysis", "Visual analysis of the dataset features."
            )  # noqa: F841

            # Add Numeric Features subsection
            numeric_subsection = self.report_generator.add_subsection(
                "Numeric Features"
            )  # noqa: F841
            # Add distribution plots for numeric features
            for feature in self.dataset_summary["numeric_features"]:
                self.report_generator.add_figure(
                    self._get_fig_path(f"dist_{feature}.png"),
                    f"Distribution of {feature}",
                )

            # Add Categorical Features subsection
            cat_subsection = self.report_generator.add_subsection(
                "Categorical Features"
            )  # noqa: F841
            # Add bar plots for categorical features
            for feature in self.dataset_summary["categorical_features"]:
                self.report_generator.add_figure(
                    self._get_fig_path(f"cat_{feature}.png"),
                    f"Distribution of {feature}",
                )

            # Add Correlation Analysis subsection
            corr_subsection = self.report_generator.add_subsection(
                "Correlation Analysis"
            )  # noqa: F841
            self.report_generator.add_figure(
                self._get_fig_path("correlation_matrix.png"),
                "Feature Correlation Matrix",
            )
            self.report_generator.add_figure(
                self._get_fig_path("target_correlations.png"),
                "Feature Correlations with Target",
            )

            # Add Model Performance section
            performance_section = self.report_generator.add_section(
                "Model Performance"
            )  # noqa: F841

            # Add Cross-validation Results subsection
            cv_subsection = self.report_generator.add_subsection(
                "Cross-validation Results"
            )
            cv_subsection.append(
                NoEscape(
                    f"5-fold CV Score: {cv_scores.mean():.4f} "
                    + f"(+/- {cv_scores.std() * 2:.4f})"
                )
            )

            # Add Test Set Performance subsection
            test_subsection = self.report_generator.add_subsection(
                "Test Set Performance"
            )  # noqa: F841
            self._format_classification_report(classification_report(y_test, y_pred))

            # Add Confusion Matrix subsection
            cm_subsection = self.report_generator.add_subsection(
                "Confusion Matrix"
            )  # noqa: F841
            self.report_generator.add_figure(
                self._get_fig_path("confusion_matrix.png"),
                "Normalized Confusion Matrix",
            )

            # Add Feature Importance section if available
            if hasattr(classifier, "feature_importances_"):
                importance_section = self.report_generator.add_section(
                    "Feature Importance Analysis"
                )
                importance_section.append(
                    "Analysis of feature importance in the model."
                )
                self.report_generator.add_figure(
                    self._get_fig_path("feature_importance.png"),
                    "Feature Importance Rankings",
                )

            # Generate final PDF
            logger.info("Generating final PDF report")
            self.report_generator.generate(f"{self._output_dir}/analysis_report")
            logger.info("Report generation complete")

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

    def _dict_to_latex_table(self, *args, **kwargs) -> None:
        """Create a LaTeX table from dictionary."""
        self.report_generator.add_table(*args, **kwargs)

    def _format_classification_report(self, report: str) -> str:
        """Format classification report for LaTeX.

        Args:
            report (str): Classification report string.

        Returns:
            str: Formatted report.
        """
        self.report_generator.add_verbatim(report)

    def _get_fig_path(self, name: str) -> str:
        """
        Returns formatted figure path.

        Args:
            name (str): figure name that will be added to `obj`:`self._figure_dir`
        """
        return os.path.join(self._figure_dir, name)
