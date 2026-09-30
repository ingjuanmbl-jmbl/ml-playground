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

The project contains package boundaries, dataset handling, generic training and evaluation, and a
Streamlit application for exploration and classification.

## Classification evaluation

`evaluate_classification(TrainingOutput)` calculates held-out metrics only. Precision, recall, and
F1 use macro averaging, giving each class equal weight; undefined class-level divisions use
`zero_division=0`. Binary ROC-AUC accepts positive-class probabilities or decision scores.
Multiclass ROC-AUC uses one-vs-rest with macro averaging and requires a score column for every
class. If scores are absent, malformed, or ROC-AUC is methodologically undefined (for example,
`y_test` contains one class), the metric is `None` and `metric_availability["roc_auc"]` explains why.
Confusion matrices retain their row/column order in `class_labels`.

## Logistic Regression

The initial model registry includes Logistic Regression (`logistic_regression`). Its declarative
specification exposes `C`, a pedagogical `regularization` selector (`L1`/`L2`), `solver`, and
`max_iter`. The factory maps `L1` to `l1_ratio=1` and `L2` to `l1_ratio=0`, without passing the
deprecated `penalty` argument. `elasticnet` remains unavailable in the interface. Solver
compatibility is validated from scikit-learn 1.8's support table: L1 accepts `liblinear` and `saga`,
while L2 accepts all registered solvers. The model requests `StandardScaler`, fitted inside the
pipeline after the train/test split. The Streamlit training panel creates a `TrainingRequest`, invokes
the generic runner, evaluates its held-out output, and plots signed coefficients and (only for two
original features) a decision boundary through the fitted pipeline.

The dependency floors are scikit-learn 1.8 for the `l1_ratio`-only regularization API and Streamlit
1.51 for `st.plotly_chart(width="stretch")`. These floors match APIs used by the implementation.

## Dependency source of truth

`pyproject.toml` is the canonical dependency definition. The existing `requirements.txt` appears to
be a fully pinned environment inventory: it includes packages outside this project's declared stack,
such as Flask, openpyxl, and sodapy. It is retained for now and should not be used to install ML
Playground. Do not update both files as parallel dependency lists. If a lock or deployment
requirements file is needed later, derive it from `pyproject.toml` with an explicit reproducible
workflow.

## Decision Tree Classifier

The registered classifiers are Logistic Regression (`logistic_regression`) and Decision Tree
(`decision_tree`). Both use the same `ModelSpecification`, `TrainingRequest`, generic runner,
evaluation contract, and dynamically generated hyperparameter controls. Logistic Regression exposes
`C`, L1/L2 regularization, solver, and `max_iter`; L1 maps to `l1_ratio=1` and L2 to `l1_ratio=0`,
without passing the deprecated `penalty` argument. It requests `StandardScaler` inside the pipeline,
fitted only after the train/test split, and reports signed coefficients in scaled feature space.

Decision Tree exposes `criterion`, `splitter`, `max_depth`, `min_samples_split`,
`min_samples_leaf`, and `max_features`. `max_depth=None` leaves tree growth unconstrained by depth;
a finite depth limits model complexity and can help control overfitting, though the appropriate value
depends on the dataset. Trees do not need feature scaling here because their split choices are based
on feature thresholds/order, which standardization does not meaningfully improve. Its
`feature_importances_` output is a tree-native split-based measure, distinct from Logistic Regression
coefficients. Both classifiers can display a direct decision boundary only when the dataset has
exactly two original features, and the visualization predicts through the complete fitted pipeline.

The Streamlit training panel builds one `TrainingRequest`, invokes the generic runner, evaluates
held-out predictions, and shows outputs provided by the selected model.
