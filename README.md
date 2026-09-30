# ML Playground

ML Playground is an interactive laboratory for exploring datasets and machine learning models.
The project is organized as a Python package under `src/ml_playground`; Streamlit is the initial
application interface.

## Requirements

- Python 3.11 or newer
- Git

## Install

From the repository root, create and activate a virtual environment, then install the package and
development dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
python -m pytest
```

The initial project foundation contains package boundaries and an application startup check.
Dataset handling and machine learning functionality will be added incrementally.

## Dependency source of truth

`pyproject.toml` is the canonical dependency definition. The existing `requirements.txt` appears to
be a fully pinned environment inventory: it includes packages outside this project's declared stack,
such as Flask, openpyxl, and sodapy. It is retained for now and should not be used to install ML
Playground. Do not update both files as parallel dependency lists. If a lock or deployment
requirements file is needed later, derive it from `pyproject.toml` with an explicit reproducible
workflow.
