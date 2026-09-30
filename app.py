"""Streamlit entry point for the ML Playground dataset explorer."""

import streamlit as st
from ml_playground.ui.dataset_explorer import render_dataset_explorer


def main() -> None:
    """Render the initial dataset exploration view."""
    render_dataset_explorer()


if __name__ == "__main__":
    main()
