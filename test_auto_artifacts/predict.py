#!/usr/bin/env python3
"""
predict.py - Script de predicción del Multilayer Perceptron

REQUERIDO POR EL SUBJECT:
    - Cargar modelo entrenado
    - Realizar predicciones en un dataset
    - Evaluar usando binary cross-entropy
    - Mostrar métricas de rendimiento

Usage:
    python3 predict.py --dataset data_validation.csv --model model_weights.npy
    
    Options:
        --dataset FILE      Dataset para predicción
        --model FILE        Modelo entrenado
        --output FILE       Guardar predicciones en archivo
        --verbose           Mostrar información detallada
"""

import sys
import argparse
import json
from preprocessing import DataPreprocessor
from network import Network
from layers import DenseLayer
from metrics import print_evaluation_report, compute_all_metrics
from loss import binary_crossentropy
from config import *


def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Realiza predicciones con un modelo Multilayer Perceptron entrenado',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Archivo CSV con datos para predicción'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_FILE,
        help=f'Archivo del modelo entrenado (default: {MODEL_FILE})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Archivo donde guardar predicciones (opcional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    return parser.parse_args()


def load_model_from_file(filepath):
    """
    Carga un modelo desde archivo.
    
    Args:
        filepath: Ruta al archivo del modelo
        
    Returns:
        Tupla (network, preprocessor_config)
    """
    try:
        with open(filepath, 'r') as f:
            model_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo del modelo '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error al parsear el modelo: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extraer datos
    architecture = model_data['architecture']
    activations = model_data['activations']
    weights = model_data['weights']
    biases = model_data['biases']
    preprocessing = model_data['preprocessing']
    config = model_data.get('config', {})
    
    # Recrear red
    network = Network(
        loss=config.get('loss_function', 'categorical_crossentropy'),
        learning_rate=config.get('learning_rate', 0.01)
    )
    
    network.build(
        architecture=architecture,
        activations=activations,
        weight_init='he',  # No importa, vamos a sobrescribir
        seed=None
    )
    
    # Cargar pesos y biases
    for layer, W, b in zip(network.layers, weights, biases):
        layer.weights = W
        layer.bias = b
    
    return network, preprocessing


def load_and_preprocess_with_config(filepath, preprocessing_config):
    """
    Carga y preprocesa datos usando configuración guardada.
    
    Args:
        filepath: Ruta al archivo CSV
        preprocessing_config: Configuración del preprocesador
        
    Returns:
        Tupla (X, y, ids)
    """
    # Crear preprocesador con configuración guardada
    preprocessor = DataPreprocessor(normalization=preprocessing_config['method'])
    preprocessor.feature_means = preprocessing_config['feature_means']
    preprocessor.feature_stds = preprocessing_config['feature_stds']
    preprocessor.feature_mins = preprocessing_config['feature_mins']
    preprocessor.feature_maxs = preprocessing_config['feature_maxs']
    preprocessor.fitted = True
    
    # Cargar datos
    features, labels, ids = preprocessor.load_data(filepath)
    
    # Codificar labels
    y = preprocessor.encode_labels(labels)
    
    # Transformar features
    X = preprocessor.transform(features)
    
    return X, y, ids


def save_predictions(filepath, ids, y_true, y_pred, probabilities):
    """
    Guarda las predicciones en un archivo CSV.
    
    Args:
        filepath: Ruta del archivo de salida
        ids: IDs de las muestras
        y_true: Labels verdaderos
        y_pred: Clases predichas
        probabilities: Probabilidades predichas
    """
    try:
        with open(filepath, 'w') as f:
            f.write("id,true_label,predicted_label,probability_benign,probability_malignant\n")
            
            for id_val, yt, yp, prob in zip(ids, y_true, y_pred, probabilities):
                true_label = 'M' if yt == 1 else 'B'
                pred_label = 'M' if yp == 1 else 'B'
                
                if isinstance(prob, list):
                    prob_b, prob_m = prob[0], prob[1]
                else:
                    prob_b, prob_m = 1 - prob, prob
                
                f.write(f"{id_val},{true_label},{pred_label},{prob_b:.6f},{prob_m:.6f}\n")
        
        print(f"✅ Predicciones guardadas en '{filepath}'")
    except Exception as e:
        print(f"❌ Error al guardar predicciones: {e}", file=sys.stderr)


def main():
    """
    Función principal de predicción.
    """
    # Parsear argumentos
    args = parse_arguments()
    
    print("=" * 80)
    print("🔮 PREDICCIÓN CON MULTILAYER PERCEPTRON")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 1. CARGAR MODELO
    # ========================================================================
    
    print(f"📂 Cargando modelo desde '{args.model}'...")
    network, preprocessing_config = load_model_from_file(args.model)
    
    if args.verbose:
        network.summary()
    else:
        print(f"   ✓ Modelo cargado correctamente")
        print(f"   ✓ Parámetros: {network.get_parameters_count()}")
        print()
    
    # ========================================================================
    # 2. CARGAR Y PREPROCESAR DATOS
    # ========================================================================
    
    print(f"📊 Cargando dataset desde '{args.dataset}'...")
    X, y, ids = load_and_preprocess_with_config(args.dataset, preprocessing_config)
    print(f"   ✓ Muestras cargadas: {len(X)}")
    print(f"   ✓ Features por muestra: {len(X[0])}")
    print()
    
    # ========================================================================
    # 3. REALIZAR PREDICCIONES
    # ========================================================================
    
    print("🎯 Realizando predicciones...")
    predictions = network.predict(X)
    print(f"   ✓ Predicciones completadas para {len(predictions)} muestras")
    print()
    
    # Convertir a clases
    y_pred_classes = []
    for pred in predictions:
        if isinstance(pred, list):
            pred_class = pred.index(max(pred))
        else:
            pred_class = 1 if pred >= 0.5 else 0
        y_pred_classes.append(pred_class)
    
    # ========================================================================
    # 4. EVALUAR CON BINARY CROSS-ENTROPY (requerido por subject)
    # ========================================================================
    
    print("📏 Evaluación con Binary Cross-Entropy (según subject):")
    print()
    
    # Extraer probabilidad de clase positiva para binary cross-entropy
    if isinstance(predictions[0], list):
        # One-hot output: tomar probabilidad de clase 1
        y_pred_probs = [pred[1] for pred in predictions]
    else:
        y_pred_probs = predictions
    
    # Calcular binary cross-entropy (fórmula del subject)
    bce_loss = binary_crossentropy(y, y_pred_probs)
    print(f"   Binary Cross-Entropy: {bce_loss:.6f}")
    print()
    
    # Formula del subject para referencia
    print("   Formula (del subject):")
    print("   E = -1/N × Σ[yn log(pn) + (1-yn) log(1-pn)]")
    print()
    
    # ========================================================================
    # 5. MÉTRICAS DETALLADAS
    # ========================================================================
    
    print_evaluation_report(y, predictions)
    
    # ========================================================================
    # 6. GUARDAR PREDICCIONES (opcional)
    # ========================================================================
    
    if args.output:
        print("💾 Guardando predicciones...")
        save_predictions(args.output, ids, y, y_pred_classes, predictions)
        print()
    
    # ========================================================================
    # 7. EJEMPLOS DE PREDICCIONES
    # ========================================================================
    
    if args.verbose and len(predictions) > 0:
        print("🔍 Ejemplos de predicciones:")
        print()
        print(f"{'ID':<12} {'Real':<8} {'Pred':<8} {'Prob B':<10} {'Prob M':<10} {'Correct':<8}")
        print("-" * 80)
        
        # Mostrar primeras 10 predicciones
        for i in range(min(10, len(predictions))):
            id_val = ids[i] if i < len(ids) else 'N/A'
            true_class = y[i]
            pred_class = y_pred_classes[i]
            
            if isinstance(predictions[i], list):
                prob_b, prob_m = predictions[i][0], predictions[i][1]
            else:
                prob_b, prob_m = 1 - predictions[i], predictions[i]
            
            true_label = 'Maligno' if true_class == 1 else 'Benigno'
            pred_label = 'Maligno' if pred_class == 1 else 'Benigno'
            correct = '✓' if true_class == pred_class else '✗'
            
            print(f"{id_val:<12} {true_label:<8} {pred_label:<8} "
                  f"{prob_b:<10.4f} {prob_m:<10.4f} {correct:<8}")
        
        if len(predictions) > 10:
            print(f"   ... ({len(predictions) - 10} muestras adicionales)")
        print()
    
    print("=" * 80)
    print("✅ PREDICCIÓN COMPLETADA")
    print("=" * 80)
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("⚠️  Predicción interrumpida por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
