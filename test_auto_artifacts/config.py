#!/usr/bin/env python3
"""
config.py - Configuración del proyecto Multilayer Perceptron

Define hiperparámetros y configuraciones por defecto del modelo.
"""

# ============================================================================
# CONFIGURACIÓN DEL DATASET
# ============================================================================

# Archivos de datos
DATASET_FILE = 'data.csv'
TRAIN_FILE = 'data_training.csv'
VALIDATION_FILE = 'data_validation.csv'

# División del dataset
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42

# Características del dataset
N_FEATURES = 30  # 30 características de células nucleares
N_CLASSES = 2    # Binario: Maligno (1) o Benigno (0)

# ============================================================================
# CONFIGURACIÓN DE LA RED NEURONAL
# ============================================================================

# Arquitectura de la red (por defecto)
# Input: 30 features → Hidden1: 24 → Hidden2: 24 → Output: 2 classes
DEFAULT_LAYERS = [30, 24, 24, 2]

# Funciones de activación
DEFAULT_HIDDEN_ACTIVATION = 'sigmoid'  # sigmoid, tanh, relu
DEFAULT_OUTPUT_ACTIVATION = 'softmax'  # softmax para clasificación

# Inicialización de pesos
DEFAULT_WEIGHT_INITIALIZER = 'he'  # he, xavier, random

# ============================================================================
# HIPERPARÁMETROS DE ENTRENAMIENTO
# ============================================================================

# Optimización
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 32  # Mini-batch gradient descent

# Early stopping
EARLY_STOPPING = True
PATIENCE = 10  # Épocas sin mejora antes de detener

# Función de pérdida
LOSS_FUNCTION = 'binary_crossentropy'

# ============================================================================
# CONFIGURACIÓN DE OUTPUT
# ============================================================================

# Archivos de salida
MODEL_FILE = 'model_weights.npy'
HISTORY_FILE = 'training_history.json'
PLOT_LOSS_FILE = 'loss_curve.png'
PLOT_ACCURACY_FILE = 'accuracy_curve.png'

# Verbosidad
VERBOSE = True
SHOW_EPOCH_METRICS = True

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================

# Tamaño de figuras
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 6

# Estilo de gráficos
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Estilo matplotlib

# ============================================================================
# LÍMITES DE SEGURIDAD
# ============================================================================

# Prevenir configuraciones problemáticas
MAX_EPOCHS = 10000
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 10.0
MIN_BATCH_SIZE = 1
MAX_LAYERS = 20

# ============================================================================
# MAPEOS
# ============================================================================

# Mapeo de etiquetas
LABEL_MAPPING = {
    'M': 1,  # Maligno
    'B': 0   # Benigno
}

LABEL_NAMES = {
    0: 'Benigno',
    1: 'Maligno'
}

# ============================================================================
# NOMBRES DE FEATURES
# ============================================================================

FEATURE_NAMES = [
    # Mean values
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    
    # Standard error
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se",
    
    # Worst values
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]


def get_config_dict():
    """
    Retorna un diccionario con toda la configuración.
    
    Returns:
        Dict con configuración
    """
    return {
        'n_features': N_FEATURES,
        'n_classes': N_CLASSES,
        'layers': DEFAULT_LAYERS,
        'hidden_activation': DEFAULT_HIDDEN_ACTIVATION,
        'output_activation': DEFAULT_OUTPUT_ACTIVATION,
        'weight_initializer': DEFAULT_WEIGHT_INITIALIZER,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'early_stopping': EARLY_STOPPING,
        'patience': PATIENCE,
        'loss_function': LOSS_FUNCTION,
        'random_seed': RANDOM_SEED
    }


def print_config(config=None):
    """
    Imprime la configuración del modelo de forma legible.
    
    Args:
        config: Dict con configuración (None usa la default)
    """
    if config is None:
        config = get_config_dict()
    
    print("⚙️  CONFIGURACIÓN DEL MODELO:")
    print("-" * 70)
    
    if 'layers' in config:
        print(f"   Arquitectura: {' → '.join(str(n) for n in config['layers'])}")
    
    for key, value in config.items():
        if key == 'layers':
            continue
        formatted_key = key.replace('_', ' ').title()
        print(f"   {formatted_key}: {value}")
    
    print()
