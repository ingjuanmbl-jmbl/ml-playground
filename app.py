"""Minimal Streamlit entry point for ML Playground."""

import streamlit as st


def main() -> None:
    """Render a simple startup confirmation."""
    st.title("ML Playground")
    st.write("Streamlit is running. The project foundation is ready.")


if __name__ == "__main__":
    main()
