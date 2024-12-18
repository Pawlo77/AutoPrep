from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class EDAVisualizer:
    """Generates comprehensive exploratory data analysis visualizations.

    This class handles the creation and saving of various data visualization plots.

    Attributes:
        output_dir (str): Directory where visualization files will be saved.
        figures (List[str]): List of generated figure paths.
    """

    def __init__(self, output_dir: str) -> None:
        """Initialize the visualizer.

        Args:
            output_dir (str): Directory for saving figures.
        """
        logger.info("Initializing EDAVisualizer with output_dir: %s", output_dir)
        self.output_dir = output_dir
        self.figures: List[str] = []

    def generate_distribution_plots(
        self, df: pd.DataFrame, features: List[str]
    ) -> List[str]:
        """Generate distribution plots for numeric features.

        Args:
            df (pd.DataFrame): Input dataset.
            features (List[str]): List of feature names to plot.

        Returns:
            List[str]: Paths to generated plot files.
        """
        logger.info("Generating distribution plots for %d features", len(features))
        figure_paths = []

        for feature in features:
            try:
                logger.debug("Creating distribution plot for feature: %s", feature)
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=feature, kde=True)
                plt.title(f"Distribution of {feature}")

                path = f"{self.output_dir}/dist_{feature}.png"
                plt.savefig(path)
                plt.close()
                figure_paths.append(path)
                logger.debug("Saved distribution plot to: %s", path)
            except Exception as e:
                logger.error(
                    "Failed to generate distribution plot for %s: %s", feature, str(e)
                )

        return figure_paths

    def generate_correlation_matrix(self, df: pd.DataFrame) -> str:
        """
        Generates a heatmap of the correlation matrix for the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame for which to generate the
                correlation matrix.

        Returns:
            str: The path to the saved correlation matrix heatmap file.
        """
        logger.start_operation("correlation matrix generation")
        try:
            plt.figure(figsize=(12, 8))
            numeric_df = df.select_dtypes(include=["int64", "float64"])
            logger.debug(
                f"Calculating correlations for {len(numeric_df.columns)} "
                "numeric features"
            )

            sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
            plt.title("Feature Correlation Matrix")

            path = f"{self.output_dir}/correlation_matrix.png"
            plt.savefig(path)
            plt.close()
            logger.debug(f"Saved correlation matrix to {path}")
            logger.end_operation()
            return path
        except Exception as e:
            logger.error(f"Failed to generate correlation matrix: {str(e)}")
            raise

    def generate_missing_values_plot(self, df: pd.DataFrame) -> str:
        """
        Generates a plot to visualize the percentage of missing values for each
        feature in the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame for which to generate the
                missing values plot.

        Returns:
            str: The path to the saved missing values plot file.
        """
        logger.start_operation("missing values visualization")
        try:
            plt.figure(figsize=(10, 6))
            missing = df.isnull().sum() / len(df) * 100
            missing = missing[missing > 0].sort_values(ascending=False)

            if missing.empty:
                logger.info("No missing values found in the dataset")
                logger.end_operation()
                return ""

            logger.debug(f"Plotting missing values for {len(missing)} features")
            sns.barplot(x=missing.index, y=missing.values)
            plt.xticks(rotation=45)
            plt.title("Percentage of Missing Values by Feature")

            path = f"{self.output_dir}/missing_values.png"
            plt.savefig(path)
            plt.close()
            logger.debug(f"Saved missing values plot to {path}")
            logger.end_operation()
            return path
        except Exception as e:
            logger.error(f"Failed to generate missing values plot: {str(e)}")
            raise

    def generate_target_correlations(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Generate correlation plot between features and target.

        Args:
            X (pd.DataFrame): Input features (already encoded/transformed).
            y (pd.Series): Target variable.

        Returns:
            str: Path to generated plot.
        """
        if X.empty:
            print("Warning: No features found for correlation analysis")
            return ""

        # Combine features with target
        data = X.copy()
        y = y.copy()
        y.name = "target"
        data = pd.concat([data, y], axis=1)

        # Calculate correlations with numeric_only=True to avoid warnings
        correlations = data.corr(numeric_only=True)["target"].sort_values(
            ascending=False
        )
        correlations = correlations.drop("target")

        if correlations.empty:
            print("Warning: No correlations found with target")
            return ""

        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlations.index, y=correlations.values)
        plt.xticks(rotation=45, ha="right")
        plt.title("Feature Correlations with Target")
        plt.tight_layout()

        path = f"{self.output_dir}/target_correlations.png"
        plt.savefig(path)
        plt.close()
        return path

    def generate_target_distribution(self, y: pd.Series) -> str:
        """
        Generates a plot showing the distribution of the target variable.

        Args:
            y (pd.Series): The target variable as a pandas Series.

        Returns:
            str: The path to the generated plot file.
        """
        logger.start_operation("target distribution plot")
        try:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=y)
            plt.title("Target Variable Distribution")

            path = f"{self.output_dir}/target_distribution.png"
            plt.savefig(path)
            plt.close()
            logger.debug(f"Saved target distribution plot to {path}")
            logger.end_operation()
            return path
        except Exception as e:
            logger.error(f"Failed to generate target distribution plot: {str(e)}")
            raise

    def generate_categorical_plots(
        self, df: pd.DataFrame, features: List[str]
    ) -> List[str]:
        """
        Generates bar plots for categorical features in the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the categorical features.
            features (List[str]): A list of categorical feature names
                to generate plots for.

        Returns:
            List[str]: A list of paths to the generated plot files.
        """
        logger.start_operation("categorical features plots")
        paths = []
        try:
            for feature in features:
                plt.figure(figsize=(10, 6))
                value_counts = df[feature].value_counts()
                if len(value_counts) > 10:
                    # Show only top 10 categories for readability
                    value_counts = value_counts.head(10)
                    plt.title(f"Top 10 Categories in {feature}")
                else:
                    plt.title(f"Distribution of {feature}")

                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                path = f"{self.output_dir}/cat_{feature}.png"
                plt.savefig(path)
                plt.close()
                paths.append(path)
                logger.debug(f"Generated plot for {feature}")

            logger.end_operation()
            return paths
        except Exception as e:
            logger.error(f"Failed to generate categorical plots: {str(e)}")
            raise
