"""Script to run all checks."""

import subprocess


def main():
    """Run all checks."""
    print("Running formatters...")
    subprocess.run(["poetry", "run", "format"], check=True)

    print("\nRunning linters...")
    subprocess.run(["poetry", "run", "lint"], check=True)


if __name__ == "__main__":
    main()
