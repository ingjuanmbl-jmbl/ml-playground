"""Streamlit entry point for dataset exploration and model training."""

import streamlit as st
from ml_playground.ui.dataset_explorer import render_dataset_explorer


def main() -> None:
    """Render the dataset explorer and registered model workflow."""
    render_dataset_explorer()


if __name__ == "__main__":
    main()
