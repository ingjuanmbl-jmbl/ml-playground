# 🧪 ML Playground

ML Playground es un laboratorio interactivo para explorar datasets, modelos de Machine Learning e
hiperparámetros. Permite entrenar modelos, analizar sus métricas y visualizar cómo cambian los
resultados al modificar una configuración. La aplicación está organizada como un paquete de Python
en `src/ml_playground` y utiliza Streamlit como interfaz.

## Requisitos

- Python 3.14.3 para reproducir el entorno probado.
- Git.

## Instalación

Desde la raíz del repositorio, crea un entorno virtual e instala el paquete y sus dependencias de
desarrollo. En Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

En macOS o Linux, activa el entorno con `source .venv/bin/activate` después de crearlo con
`python3.14 -m venv .venv`. Las versiones principales están fijadas en `pyproject.toml`, que es la
fuente de verdad de dependencias. Para configurar o comprobar el entorno desde PowerShell también
puedes usar `scripts/setup.ps1`.

## Ejecución y pruebas

```powershell
python -m streamlit run app.py
python -m pytest -q
python -m pip check
```

## Cómo usar el laboratorio

1. Elige un conjunto de datos y revisa sus variables y su objetivo.
2. Abre la sección **Modelos** y selecciona un estimador compatible.
3. Revisa la explicación del modelo y ajusta los hiperparámetros disponibles.
4. Ejecuta el entrenamiento y examina métricas y gráficos.
5. Usa el explorador de hiperparámetros para observar el efecto de cambiar un parámetro.
6. Compara varios clasificadores sobre una partición común.

## Datasets

Se incluyen Iris, Wine, Breast Cancer y Digits, además de los generadores sintéticos
`make_classification`, `make_moons`, `make_circles`, `make_blobs` y `make_regression`. Los parámetros
de generación y la semilla (`random_state`) quedan registrados para poder reproducir los datos.
En un dataset sintético, `n_samples` determina cuántas observaciones se generan y `noise` agrega
variación cuando ese generador lo admite.

## Modelos de clasificación

Los clasificadores Logistic Regression, Decision Tree, Random Forest, MLPClassifier y XGBoost usan
las mismas especificaciones declarativas, el registro común, el pipeline de preprocessing, el
Generic Training Runner y el módulo Evaluation. Sus controles de hiperparámetros se generan desde
las especificaciones. Los modelos que requieren escalamiento lo ajustan dentro del pipeline y
únicamente con los datos de entrenamiento.

### Logistic Regression

Logistic Regression es un clasificador lineal que estima probabilidades y separa clases mediante
una frontera lineal. Sus controles incluyen `C`, `regularization` (L1 o L2), `solver` y `max_iter`.
Un valor menor de `C` aplica una regularización más fuerte; L1 puede producir coeficientes nulos,
mientras L2 los reduce gradualmente. La implementación usa la API actual de scikit-learn mediante
`l1_ratio` y no pasa el parámetro deprecado `penalty`. El modelo utiliza `StandardScaler` dentro del
pipeline y muestra coeficientes con signo, no una importancia de variables.

### Decision Tree

Decision Tree crea reglas mediante particiones sucesivas de las variables. Expone `criterion`,
`splitter`, `max_depth`, `min_samples_split`, `min_samples_leaf` y `max_features`. `max_depth=None`
no impone un límite de profundidad; un límite finito restringe el tamaño de las reglas y puede
ayudar a controlar la complejidad. Los árboles no necesitan escalamiento en este flujo porque sus
particiones se basan en umbrales de las variables. El estimador puede mostrar su medida nativa
`feature_importances_`, que no representa causalidad.

### Random Forest

Random Forest combina las predicciones de varios árboles. `n_estimators` controla cuántos árboles
se ajustan; también se ofrecen `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
y `random_state`. El modelo no requiere escalamiento y puede presentar los valores
`feature_importances_` nativos del estimador. Aumentar la cantidad de árboles puede elevar el tiempo
de entrenamiento; ninguna configuración se considera universalmente superior.

### XGBoost

XGBoost construye árboles de manera secuencial (boosting), agregando etapas al ensamble existente.
`n_estimators` indica las etapas, `learning_rate` escala cada actualización y `max_depth` limita la
profundidad de los árboles. También se exponen parámetros de muestreo y regularización. El modelo
no utiliza escalamiento y muestra `feature_importances_` del estimador; sus valores no son
coeficientes ni una explicación causal.

### MLPClassifier

MLPClassifier es una red neuronal multicapa. `hidden_layer_sizes` indica el número de neuronas de
cada capa oculta: `(10,)` representa una capa con 10 neuronas y `(20, 10)` representa dos capas.
`alpha` controla la regularización L2, `learning_rate_init` define el paso inicial del optimizador,
`max_iter` limita las iteraciones y `early_stopping` reserva una porción de los datos de
entrenamiento para vigilar la mejora. Se muestra la curva `loss_curve_` cuando está disponible.
MLPClassifier utiliza `StandardScaler` dentro del pipeline. Si aparece un aviso de convergencia,
significa que se alcanzó el límite de iteraciones antes del criterio de parada.

## K-Means y clustering

K-Means agrupa observaciones según su distancia a centroides y no utiliza las etiquetas objetivo
durante el entrenamiento. `n_clusters` indica cuántos grupos se solicitan. El modelo usa
`StandardScaler` dentro del pipeline porque la distancia y la posición de los centroides dependen
de la escala de las variables. La interfaz muestra centroides, gráfico y métricas internas:
Silhouette, Davies-Bouldin y Calinski-Harabasz. Cada métrica describe un aspecto de la estructura;
ninguna determina por sí sola la calidad de una segmentación.

## Evaluación de clasificación

Evaluation opera sobre los resultados del conjunto de prueba y no vuelve a entrenar el modelo.
Precision, Recall y F1 usan promedio macro para dar el mismo peso a cada clase; las divisiones
indefinidas usan `zero_division=0`. Para clasificación binaria, ROC-AUC acepta probabilidades o
scores de decisión. Para multiclase usa uno contra el resto con promedio macro y requiere un score
por clase. Si no hay scores válidos o ROC-AUC no se puede calcular metodológicamente, el resultado
es `None` y se conserva el motivo. La matriz de confusión mantiene el orden de clases en sus ejes.

## Explorador de hiperparámetros

El explorador modifica un hiperparámetro por vez y reutiliza dataset, semilla y partición cuando
corresponde. Es una exploración de sensibilidad, no una búsqueda automática ni un mecanismo de
selección del mejor modelo. Sus resultados dependen de los datos y de la partición; no constituyen
por sí solos una evaluación definitiva de generalización.

## Comparación de modelos

La comparación ejecuta dos o más clasificadores sobre los mismos datos, objetivo y partición
train/test. Informa Exactitud, Precisión, Sensibilidad, F1, ROC-AUC cuando esté disponible y tiempo
de entrenamiento. Los gráficos conservan el orden seleccionado y no generan ganador, ranking ni
recomendación. Un único split facilita una comparación lado a lado, pero no constituye una
evaluación exhaustiva de generalización ni sustituye validación cruzada o datos independientes.

## Dependencias

`pyproject.toml` es la única fuente declarativa de dependencias del proyecto y fija las versiones
principales probadas: NumPy 2.4.3, pandas 2.3.3, Plotly 6.6.0, scikit-learn 1.8.0, Streamlit 1.55.0
y XGBoost 3.4.1. No uses `requirements.txt` para instalar el proyecto ni lo mantengas como una
segunda lista de dependencias.
