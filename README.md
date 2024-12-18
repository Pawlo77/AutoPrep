# Auto-Prep

Auto-Prep is an automated data preprocessing and analysis pipeline that generates comprehensive LaTeX reports. It handles common preprocessing tasks, creates insightful visualizations, and documents the entire process in a professional PDF report. It focuses on tabluar data, supporting numerous explainable AI models. Focusing on their interpretability and ease of use, it includes subsections for each model, explaining their strengths and weaknesses and providing examples of their usage.

## Features

- Automated data cleaning and preprocessing
- Intelligent feature type detection
- Advanced categorical encoding with rare category handling
- Comprehensive exploratory data analysis (EDA)
- Automated visualization generation
- Professional LaTeX report generation
- Modular and extensible design
- Support for numerous explainable AI models
- Explainability
- Examples of usage

## Installation

### Using Poetry (Recommended)

1. Make sure you have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/auto-prep.git
cd auto-prep
```

3. Install dependencies:

```bash
poetry install
```

4. Activate the virtual environment:

```bash
poetry shell
```

5. Run the example usage:

```bash
python example_usage.py
```

## Output Structure

The pipeline generates the following output structure:

reports/
├── analysis_report.pdf # Main LaTeX report
└── figures/ # Generated visualizations
├── correlation_matrix.png
├── missing_values.png
└── dist_.png # Distribution plots for numeric features

## Report Contents

The generated report includes:
1. Title page and table of contents
2. Data Overview
   - Dataset structure
   - Feature characteristics
3. Exploratory Data Analysis
   - Distribution plots
   - Correlation matrix
   - Missing value analysis
4. Model Performance
   - Accuracy metrics
   - Model details

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Paweł Pozorski - [GitHub](https://github.com/pozorski)

## Acknowledgments

- Inspired by the need for automated preprocessing and reporting in data science workflows
- Built with modern Python tools and best practices

## Notes for developers

1. Poetry is used for dependency management and virtual environment. Following functions are implemented:
- `poetry install` - install dependencies
- `poetry shell` - activate virtual environment
- `poetry run format` - format code
- `poetry run lint` - lint code
- `poetry run check` - check code
- `poetry run test` - run tests
- `poetry run pre-commit install` - install pre-commit hooks
- `poetry run pre-commit run --all-files` - run pre-commit hooks
