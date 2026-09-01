"""Entry point for the customer segmentation project."""

from pathlib import Path


def main():
    """Run the customer segmentation workflow."""
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")
    print("Customer segmentation pipeline initialized.")


if __name__ == "__main__":
    main()
