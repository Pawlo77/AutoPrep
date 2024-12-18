"""Script to format code using black and isort."""

import subprocess
from pathlib import Path


def run_command(cmd: list, description: str, capture_output: bool = True) -> bool:
    """Run a command and handle its output.

    Args:
        cmd (list): Command to run as a list of strings.
        description (str): Description of the command for output.
        capture_output (bool): Whether to capture command output. Defaults to True.

    Returns:
        bool: True if command succeeded, False otherwise.
    """
    print(f"\nRunning {description}...")
    try:
        if capture_output:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            subprocess.run(cmd, check=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed with error:")
        if capture_output:
            print(e.stderr)
        return False


def main() -> None:
    """Run formatters."""
    project_root = Path(__file__).parent.parent
    python_files = list(project_root.glob("auto_prep/**/*.py"))

    if not python_files:
        print("No Python files found to format")
        return

    # Run black
    black_success = run_command(
        ["black"] + [str(f) for f in python_files],
        "black formatter",
        capture_output=False,
    )

    # Run isort
    isort_success = run_command(
        ["isort"] + [str(f) for f in python_files],
        "isort",
        capture_output=False,
    )

    # Run autoflake
    autoflake_success = run_command(
        [
            "autoflake",
            "--in-place",
            "--recursive",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--expand-star-imports",
            "--ignore-init-module-imports",
            "--check",
        ]
        + [str(f) for f in python_files],
        "autoflake",
        capture_output=False,
    )

    # Report overall status
    if all([black_success, isort_success, autoflake_success]):
        print("\nAll formatting completed successfully")
    else:
        print("\nSome formatting steps failed")
        exit(1)


if __name__ == "__main__":
    main()
