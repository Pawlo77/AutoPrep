"""Script to run linting checks."""

import subprocess
from pathlib import Path


def main() -> None:
    """Run linters."""
    project_root = Path(__file__).parent.parent
    python_files = list(project_root.glob("auto_prep/**/*.py"))

    if not python_files:
        print("No Python files found to lint")
        return

    print("Running flake8...")
    subprocess.run(
        ["flake8", "--config=.flake8"] + [str(f) for f in python_files], check=True
    )


if __name__ == "__main__":
    main()
