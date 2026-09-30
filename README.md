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

The model registry includes Logistic Regression (`logistic_regression`). Its declarative
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

The registered classifiers are Logistic Regression (`logistic_regression`), Decision Tree
(`decision_tree`), Random Forest (`random_forest`), MLPClassifier (`mlp_classifier`), and XGBoost
(`xgboost_classifier`). They use
the same `ModelSpecification`, `TrainingRequest`, generic runner,
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
coefficients. Classifiers can display a direct decision boundary only when the dataset has
exactly two original features, and the visualization predicts through the complete fitted pipeline.

The Streamlit training panel builds one `TrainingRequest`, invokes the generic runner, evaluates
held-out predictions, and shows outputs provided by the selected model.

## Random Forest Classifier

Random Forest (`random_forest`) is a registered classifier. Its declarative controls are
`n_estimators`, `max_depth` (including `None`), `min_samples_split`, `min_samples_leaf`,
`max_features`, and `random_state`. Like Decision Tree, the forest does not use feature scaling and
reports the estimator's native `feature_importances_` values. It runs through the same pipeline,
generic training runner, evaluation, and visualization flow; no algorithm-specific runner logic is
needed. `n_estimators` controls the number of trees and has a practical effect on training time.

## MLPClassifier

`MLPClassifier` (`mlp_classifier`) adds a feed-forward neural network to the same classifier registry.
`hidden_layer_sizes` is a tuple containing the neuron count in each hidden layer: `(10,)` is one
layer with 10 neurons, while `(20, 10)` is two layers with 20 and 10 neurons. The available presets
also include `(20,)` and `(50, 25, 10)`.

`alpha` controls L2 regularization of network weights. `learning_rate_init` sets the optimizer's
initial step size. `max_iter` caps training iterations; a convergence warning means the optimizer
reached that limit before satisfying its stopping criterion, so inspect the loss curve and consider
changing the iteration limit or other settings. `early_stopping` reserves a validation portion of the
training data and stops when the validation score no longer improves. `loss_curve_` is plotted only
when the fitted estimator provides it.

MLPClassifier requests `StandardScaler` inside the pipeline because its optimization is sensitive to
feature scales. The scaler is fitted on training data only. The interface reports configured layer
sizes, the approximate count of learned weights and biases, activation, iterations, and any available
early-stopping validation score. These controls are for experimentation; no architecture is assumed
to be best for every dataset.

## K-Means Clustering

K-Means (`kmeans`) is the project's unsupervised model: unlike supervised classifiers, it groups
observations from their feature values and does not use a target label. `n_clusters` sets the number
of groups requested; it is a modeling choice, not a label discovered automatically. The model uses
`StandardScaler` inside its pipeline because Euclidean distances and centroid locations are scale
sensitive. Labels from Iris, Wine, moons, and circles may be present for dataset reference, but
training and clustering metrics use only `X`.

Evaluation reports Silhouette (larger is generally more separated), Davies-Bouldin (smaller is
generally more compact and separated), and Calinski-Harabasz (larger compares between-cluster to
within-cluster dispersion). Each metric has assumptions and responds to data geometry; none alone
establishes the absolute quality or usefulness of a segmentation. A metric that is undefined for a
solution is displayed as unavailable with its reason. Centroids are inverse-transformed through the
fitted pipeline and shown in original feature units. Cluster charts display two selected original
features; when there are more features, selecting axes is not a projection of the full feature space.

## XGBoost Classifier

XGBoost (`xgboost_classifier`) adds gradient-boosted decision trees through the same registry and
training flow. Unlike Random Forest, which fits trees as an ensemble and aggregates their outputs,
boosting adds trees sequentially to improve the current ensemble. The interface exposes a focused
set of controls: `n_estimators` is the number of boosting rounds, `max_depth` limits the depth of
each tree, and `learning_rate` scales each update. It also exposes child weight, row and feature
sampling, split threshold, and L1/L2 leaf-weight regularization. The wrapper determines a suitable
classification objective from the target, including binary and multiclass cases.

XGBoost does not request feature scaling. The feature importance display uses the estimator's native
`feature_importances_` values; these are model-derived values, not coefficients or a causal measure.
