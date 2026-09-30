"""Spanish educational copy used by the Streamlit presentation layer."""

from __future__ import annotations

from ml_playground.models.specifications import ProblemType


MODEL_GUIDES: dict[str, tuple[str, str]] = {
    "logistic_regression": (
        "Es un clasificador lineal: estima probabilidades y separa las clases mediante una frontera lineal. "
        "La regularización controla el tamaño de sus coeficientes. C pequeño implica regularización más fuerte; "
        "L1 puede llevar algunos coeficientes a cero y L2 los reduce de forma gradual. max_iter limita las iteraciones.",
        "Puede ser útil como referencia interpretable cuando una frontera aproximadamente lineal es razonable. "
        "El resultado depende de los datos, la escala y la regularización elegida.",
    ),
    "decision_tree": (
        "Construye reglas mediante particiones sucesivas de las variables. max_depth limita cuántas particiones "
        "puede encadenar; min_samples_split exige suficientes observaciones para dividir un nodo y "
        "min_samples_leaf establece el mínimo en cada hoja.",
        "Puede ser útil para explorar reglas no lineales y relaciones que se expresan mediante umbrales. "
        "Una profundidad grande puede producir reglas muy específicas del conjunto de entrenamiento.",
    ),
    "random_forest": (
        "Combina las predicciones de varios árboles entrenados con variaciones de observaciones y variables. "
        "n_estimators define cuántos árboles se ajustan; max_depth limita su profundidad y max_features "
        "controla el muestreo de variables en cada división. Puede mostrar la medida nativa de importancia del estimador.",
        "Puede ser útil para explorar relaciones no lineales con un conjunto de árboles. El costo de entrenamiento "
        "suele crecer al aumentar n_estimators; las importancias no indican causalidad.",
    ),
    "xgboost_classifier": (
        "XGBoost utiliza boosting: agrega árboles de forma secuencial para corregir el comportamiento del conjunto "
        "acumulado. n_estimators define las etapas, learning_rate escala cada actualización y max_depth limita "
        "la complejidad de cada árbol. reg_alpha y reg_lambda controlan regularización L1 y L2.",
        "Puede ser útil para experimentar con ensambles de árboles y ajustes graduales. Sus resultados y costo "
        "dependen del dataset y de la combinación de parámetros.",
    ),
    "mlp_classifier": (
        "Es una red neuronal multicapa. hidden_layer_sizes especifica cuántas neuronas hay en cada capa oculta; "
        "por ejemplo, (20, 10) representa dos capas. alpha controla la regularización L2, learning_rate_init "
        "es el paso inicial de optimización, max_iter limita las iteraciones y early_stopping reserva datos de "
        "entrenamiento para vigilar la convergencia.",
        "Puede ser útil para explorar fronteras no lineales. La optimización depende de la escala, la arquitectura "
        "y los parámetros; una advertencia de convergencia indica que se alcanzó el límite de iteraciones.",
    ),
    "kmeans": (
        "K-Means agrupa observaciones asignándolas al centroide más cercano y actualiza los centroides de forma "
        "iterativa. n_clusters establece cuántos grupos se solicitan. La distancia y la escala de las variables "
        "influyen en las asignaciones.",
        "Puede ser útil para explorar agrupaciones compactas cuando la distancia entre observaciones es informativa. "
        "Es sensible a la escala y al número de grupos elegido; las etiquetas de cluster no son clases reales.",
    ),
}

HYPERPARAMETER_HELP: dict[str, str] = {
    "C": "Controla la regularización inversa: valores menores aplican una penalización más fuerte.",
    "regularization": "Selecciona L1 o L2. L1 puede producir coeficientes iguales a cero; L2 los reduce gradualmente.",
    "solver": "Método numérico usado para ajustar el clasificador.",
    "max_iter": "Número máximo de iteraciones del optimizador o del algoritmo.",
    "criterion": "Criterio usado para medir la calidad de una partición del árbol.",
    "splitter": "Estrategia para elegir la partición de cada nodo.",
    "max_depth": "Profundidad máxima del árbol. Valores pequeños limitan su complejidad; None no fija un límite de profundidad.",
    "min_samples_split": "Número mínimo de observaciones que debe tener un nodo para poder dividirse.",
    "min_samples_leaf": "Número mínimo de observaciones que debe quedar en cada hoja.",
    "max_features": "Cantidad o proporción de variables consideradas al buscar una partición.",
    "n_estimators": "Número de árboles o etapas de ensamble utilizados por el modelo.",
    "random_state": "Semilla que permite reproducir los pasos aleatorios del estimador.",
    "hidden_layer_sizes": "Número de neuronas en cada capa oculta; por ejemplo, (20, 10) representa dos capas.",
    "activation": "Función de activación aplicada en las capas ocultas.",
    "alpha": "Intensidad de la regularización L2 aplicada a los pesos de la red.",
    "learning_rate": "Estrategia que determina cómo cambia la tasa de aprendizaje durante el ajuste.",
    "learning_rate_init": "Tamaño inicial de los pasos usados para actualizar los pesos.",
    "batch_size": "Cantidad de observaciones procesadas en cada actualización del optimizador.",
    "early_stopping": "Detiene el ajuste si el resultado en una porción de validación deja de mejorar.",
    "min_child_weight": "Peso mínimo de observaciones requerido en un nodo hijo de XGBoost.",
    "subsample": "Fracción de observaciones de entrenamiento muestreadas en cada etapa de boosting.",
    "colsample_bytree": "Fracción de variables muestreadas para construir cada árbol.",
    "gamma": "Reducción mínima de pérdida requerida para aceptar una partición en XGBoost.",
    "reg_alpha": "Intensidad de regularización L1 de los pesos de las hojas en XGBoost.",
    "reg_lambda": "Intensidad de regularización L2 de los pesos de las hojas en XGBoost.",
    "n_clusters": "Cantidad de grupos que K-Means intentará formar.",
    "init": "Método utilizado para elegir los centroides iniciales.",
    "n_init": "Cantidad de inicializaciones distintas; se conserva la mejor según el criterio del algoritmo.",
    "noise": "Desviación agregada a los datos sintéticos; al aumentarla, los puntos se dispersan más.",
    "n_samples": "Cantidad de observaciones que tendrá el dataset sintético.",
    "n_features": "Cantidad de variables predictoras generadas.",
    "n_informative": "Cantidad de variables generadas con información para el objetivo.",
    "n_redundant": "Cantidad de variables generadas como combinaciones redundantes de otras variables.",
    "n_repeated": "Cantidad de variables que repiten otras variables generadas.",
    "n_classes": "Cantidad de clases que tendrá el objetivo sintético.",
    "n_clusters_per_class": "Cantidad de grupos internos que se generan por clase.",
    "class_sep": "Separación entre grupos de clases en el dataset sintético.",
    "flip_y": "Fracción aproximada de etiquetas que se asigna aleatoriamente para introducir ruido.",
    "factor": "Proporción entre el radio del círculo interno y el externo.",
    "centers": "Cantidad de centroides alrededor de los cuales se generan los grupos.",
    "cluster_std": "Dispersión de las observaciones alrededor de cada centro.",
    "bias": "Término constante añadido a los valores objetivo sintéticos.",
}

DATASET_GUIDES: dict[str, str] = {
    "iris": "Mediciones de sépalos y pétalos de flores Iris con tres especies como objetivo.",
    "wine": "Análisis químicos de vinos agrupados por su clase de origen.",
    "breast_cancer": "Mediciones de núcleos celulares para distinguir diagnósticos benignos y malignos.",
    "digits": "Imágenes de dígitos escritos a mano representadas mediante variables de intensidad de píxeles.",
    "make_classification": "Genera variables y etiquetas de clasificación controlando tamaño, separación, ruido y estructura.",
    "make_moons": "Genera dos formas de media luna entrelazadas; noise agrega dispersión a las coordenadas.",
    "make_circles": "Genera dos círculos concéntricos; noise agrega dispersión y factor determina la proporción de sus radios.",
    "make_blobs": "Genera grupos de puntos alrededor de centros gaussianos para explorar clustering.",
    "make_regression": "Genera variables predictoras y un objetivo numérico con ruido configurable.",
}

DATASET_NAMES: dict[str, str] = {
    "breast_cancer": "Cáncer de mama",
    "make_classification": "Clasificación sintética",
    "make_moons": "Lunas sintéticas",
    "make_circles": "Círculos sintéticos",
    "make_blobs": "Grupos sintéticos",
    "make_regression": "Regresión sintética",
}

METRIC_LABELS: dict[str, str] = {
    "accuracy": "Exactitud (Accuracy)",
    "precision": "Precisión (Precision)",
    "recall": "Sensibilidad (Recall)",
    "f1": "Puntuación F1",
    "roc_auc": "ROC-AUC",
    "silhouette": "Coeficiente de Silhouette",
    "davies_bouldin": "Índice de Davies-Bouldin",
    "calinski_harabasz": "Índice de Calinski-Harabasz",
    "training_seconds": "Tiempo de entrenamiento (s)",
}

METRIC_HELP: dict[str, str] = {
    "accuracy": "Proporción de observaciones clasificadas correctamente.",
    "precision": "Proporción de predicciones positivas que fueron correctas; se informa el promedio macro.",
    "recall": "Proporción de positivos reales que el modelo identificó; se informa el promedio macro.",
    "f1": "Media armónica entre precisión y sensibilidad; se informa el promedio macro.",
    "roc_auc": "Capacidad de ordenar positivos sobre negativos a distintos umbrales; requiere probabilidades o scores válidos.",
    "silhouette": "Compara la cercanía al propio grupo con la distancia a otros grupos; valor interno de estructura.",
    "davies_bouldin": "Resume similitud entre grupos y su dispersión; menor valor indica grupos más separados según este índice.",
    "calinski_harabasz": "Relaciona dispersión entre grupos con dispersión dentro de los grupos.",
    "training_seconds": "Tiempo transcurrido durante el ajuste del estimador y su pipeline.",
}

PROBLEM_TYPE_LABELS = {
    ProblemType.CLASSIFICATION: "Clasificación",
    ProblemType.CLUSTERING: "Clustering (agrupamiento)",
    ProblemType.REGRESSION: "Regresión",
}


def dataset_name(dataset_id: str, fallback: str) -> str:
    if dataset_id in DATASET_NAMES:
        return DATASET_NAMES[dataset_id]
    aliases = {
        "Breast Cancer": "Cáncer de mama",
        "Synthetic Classification": "Clasificación sintética",
        "Synthetic Moons": "Lunas sintéticas",
        "Synthetic Circles": "Círculos sintéticos",
        "Synthetic Blobs": "Grupos sintéticos",
        "Synthetic Regression": "Regresión sintética",
    }
    return aliases.get(fallback, fallback)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())


__all__ = [
    "DATASET_GUIDES",
    "DATASET_NAMES",
    "HYPERPARAMETER_HELP",
    "METRIC_HELP",
    "METRIC_LABELS",
    "MODEL_GUIDES",
    "PROBLEM_TYPE_LABELS",
    "dataset_name",
    "metric_label",
]
