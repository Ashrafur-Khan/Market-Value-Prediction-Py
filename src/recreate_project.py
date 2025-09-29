"""Compatibility entrypoint for the original recreate_project script."""
from market_value.pipeline import run_pipeline


def main() -> None:
    """Execute the end-to-end pipeline while keeping legacy CLI semantics."""
    run_pipeline()


if __name__ == "__main__":
    main()
