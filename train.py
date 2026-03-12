#!/usr/bin/env python3
"""
train.py - Script principal de entrenamiento del Multilayer Perceptron

REQUERIDO POR EL SUBJECT:
    - Entrenar red con backpropagation y gradient descent
    - Mostrar métricas por época (loss, val_loss)
    - Guardar modelo entrenado
    - Generar curvas de aprendizaje (loss y accuracy)

Usage:
    python3 train.py --dataset data_training.csv [options]
    
    Options:
        --layers 24 24          Neuronas por capa oculta
        --epochs 100            Número de épocas
        --learning_rate 0.01    Learning rate
        --batch_size 32         Tamaño del batch
        --activation sigmoid    Función de activación para capas ocultas
        --validation data_validation.csv  Dataset de validación
        --seed 42               Semilla aleatoria
        --save model.npy        Archivo donde guardar el modelo

Ejemplo del subject (página 10):
    python mlp.py --dataset data_training.csv
    x_train shape : (342, 30)
    x_valid shape : (85, 30)
    epoch 01/70 - loss: 0.6882 - val_loss: 0.6788
    ...
    epoch 70/70 - loss: 0.0640 - val_loss: 0.0474
    > saving model './saved_model.npy' to disk...
"""

import sys
import argparse
import json
from preprocessing import DataPreprocessor
from network import Network
from utils import Timer, format_time
from config import *


def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Entrena un Multilayer Perceptron para clasificación de cáncer de mama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 train.py --dataset data_training.csv
  python3 train.py --dataset data_training.csv --layers 24 24 --epochs 100
  python3 train.py --dataset data_training.csv --learning_rate 0.01 --batch_size 16
        """
    )
    
    # Argumentos obligatorios
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Archivo CSV con datos de entrenamiento'
    )
    
    # Arquitectura
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[24, 24],
        help='Neuronas por capa oculta (default: 24 24)'
    )
    
    # Hiperparámetros
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Número de épocas (default: {EPOCHS})'
    )
    
    parser.add_argument(
        '--learning_rate', '--lr',
        type=float,
        default=LEARNING_RATE,
        help=f'Learning rate (default: {LEARNING_RATE})'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'Tamaño del batch (default: {BATCH_SIZE})'
    )
    
    # Configuración de la red
    parser.add_argument(
        '--activation',
        type=str,
        default=DEFAULT_HIDDEN_ACTIVATION,
        choices=['sigmoid', 'tanh', 'relu', 'leaky_relu'],
        help=f'Función de activación para capas ocultas (default: {DEFAULT_HIDDEN_ACTIVATION})'
    )
    
    parser.add_argument(
        '--weight_init',
        type=str,
        default='auto',
        choices=['auto', 'he', 'xavier', 'random'],
        help='Método de inicialización de pesos (default: auto)'
    )
    
    # Bonus: Optimizer
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        choices=['sgd', 'sgd_momentum', 'rmsprop', 'adam'],
        help='Optimizador (default: sgd, bonus: adam)'
    )
    
    # Bonus: Dropout regularization
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Tasa de dropout para capas ocultas (0.0-0.5, default: 0.0, bonus)'
    )
    
    # Datos
    parser.add_argument(
        '--validation',
        type=str,
        default=VALIDATION_FILE,
        help=f'Archivo CSV con datos de validación (default: {VALIDATION_FILE})'
    )
    
    # Early stopping (bonus)
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        default=EARLY_STOPPING,
        help='Activar early stopping'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=PATIENCE,
        help=f'Paciencia para early stopping (default: {PATIENCE})'
    )
    
    # Output
    parser.add_argument(
        '--save',
        type=str,
        default=MODEL_FILE,
        help=f'Archivo donde guardar el modelo (default: {MODEL_FILE})'
    )
    
    parser.add_argument(
        '--history',
        type=str,
        default=HISTORY_FILE,
        help=f'Archivo donde guardar historial (default: {HISTORY_FILE})'
    )
    
    # Otros
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Semilla aleatoria (default: {RANDOM_SEED})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=VERBOSE,
        help='Mostrar información detallada'
    )
    
    return parser.parse_args()


def load_and_preprocess(filepath, preprocessor=None, fit=False):
    """
    Carga y preprocesa un dataset.
    
    Args:
        filepath: Ruta al archivo CSV
        preprocessor: Instancia de DataPreprocessor (None para crear uno nuevo)
        fit: Si True, ajusta el preprocesador (solo para train data)
        
    Returns:
        Tupla (X, y, preprocessor)
    """
    if preprocessor is None:
        preprocessor = DataPreprocessor(normalization='standardization')
    
    # Cargar datos
    features, labels, ids = preprocessor.load_data(filepath)
    
    # Codificar labels
    y = preprocessor.encode_labels(labels)
    
    # Normalizar features
    if fit:
        X = preprocessor.fit_transform(features)
    else:
        X = preprocessor.transform(features)
    
    return X, y, preprocessor


def create_batches(X, y, batch_size):
    """
    Divide los datos en mini-batches.
    
    Args:
        X: Features
        y: Labels
        batch_size: Tamaño del batch
        
    Returns:
        Lista de tuplas (X_batch, y_batch)
    """
    n_samples = len(X)
    batches = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        batches.append((X_batch, y_batch))
    
    return batches


def train_epoch(network, X_train, y_train, batch_size):
    """
    Entrena una época completa.
    
    Args:
        network: Instancia de Network
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento
        batch_size: Tamaño del batch
        
    Returns:
        Pérdida promedio de la época
    """
    # Crear batches
    batches = create_batches(X_train, y_train, batch_size)
    
    epoch_loss = 0.0
    
    # Entrenar cada batch
    for X_batch, y_batch in batches:
        batch_loss = network.train_step(X_batch, y_batch)
        epoch_loss += batch_loss * len(X_batch)
    
    # Promedio ponderado
    epoch_loss /= len(X_train)
    
    return epoch_loss


def main():
    """
    Función principal de entrenamiento.
    """
    # Parsear argumentos
    args = parse_arguments()
    
    print("=" * 80)
    print("🧠 ENTRENAMIENTO DEL MULTILAYER PERCEPTRON")
    print("=" * 80)
    print()
    
    # Timer
    timer = Timer()
    timer.start()
    
    # ========================================================================
    # 1. CARGAR Y PREPROCESAR DATOS
    # ========================================================================
    
    print("📂 Cargando datos...")
    
    # Cargar datos de entrenamiento
    X_train, y_train, preprocessor = load_and_preprocess(
        args.dataset, 
        fit=True
    )
    
    print(f"   ✓ x_train shape: ({len(X_train)}, {len(X_train[0])})")
    
    # Cargar datos de validación
    try:
        X_valid, y_valid, _ = load_and_preprocess(
            args.validation,
            preprocessor=preprocessor,
            fit=False
        )
        print(f"   ✓ x_valid shape: ({len(X_valid)}, {len(X_valid[0])})")
        has_validation = True
    except:
        print(f"   ⚠️  No se pudo cargar validación, continuando solo con train")
        X_valid, y_valid = None, None
        has_validation = False
    
    print()
    
    # ========================================================================
    # 2. CREAR Y CONFIGURAR LA RED
    # ========================================================================
    
    n_features = len(X_train[0])
    n_classes = N_CLASSES
    
    # Construir arquitectura: Input → Hidden layers → Output
    architecture = [n_features] + args.layers + [n_classes]
    
    # Activaciones: hidden layers + output
    activations = [args.activation] * len(args.layers) + [DEFAULT_OUTPUT_ACTIVATION]
    
    print("🏗️  Construyendo red neuronal...")
    network = Network(
        loss='categorical_crossentropy', 
        learning_rate=args.learning_rate,
        optimizer=args.optimizer  # BONUS: Adam/RMSprop/Momentum
    )
    network.build(
        architecture=architecture,
        activations=activations,
        weight_init=args.weight_init,
        dropout_rate=args.dropout,  # BONUS: Dropout regularization
        seed=args.seed
    )
    
    if args.verbose:
        network.summary()
    else:
        arch_str = ' → '.join(str(n) for n in architecture)
        print(f"   Arquitectura: {arch_str}")
        print(f"   Parámetros totales: {network.get_parameters_count()}")
        print()
    
    # ========================================================================
    # 3. ENTRENAMIENTO
    # ========================================================================
    
    print("🎯 Iniciando entrenamiento...")
    print()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Entrenar una época
        train_loss = train_epoch(network, X_train, y_train, args.batch_size)
        
        # Evaluar en training
        _, train_acc = network.evaluate(X_train, y_train)
        
        # Evaluar en validation
        if has_validation:
            val_loss, val_acc = network.evaluate(X_valid, y_valid)
            
            # Guardar en historia
            network.history['loss'].append(train_loss)
            network.history['val_loss'].append(val_loss)
            network.history['accuracy'].append(train_acc)
            network.history['val_accuracy'].append(val_acc)
            
            # Formato del subject: "epoch 01/70 - loss: 0.6882 - val_loss: 0.6788"
            print(f"epoch {epoch:02d}/{args.epochs} - "
                  f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
                  f"acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
            
            # Early stopping
            if args.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                    if patience_counter >= args.patience:
                        print()
                        print(f"⚠️  Early stopping en época {epoch} (patience={args.patience})")
                        print(f"   Mejor val_loss: {best_val_loss:.4f}")
                        break
        else:
            # Solo training
            network.history['loss'].append(train_loss)
            network.history['accuracy'].append(train_acc)
            
            print(f"epoch {epoch:02d}/{args.epochs} - "
                  f"loss: {train_loss:.4f} - acc: {train_acc:.4f}")
    
    print()
    
    # ========================================================================
    # 4. GUARDAR MODELO
    # ========================================================================
    
    timer.stop()
    
    print("💾 Guardando modelo...")
    
    # Preparar datos del modelo
    model_data = {
        'architecture': architecture,
        'activations': activations,
        'weights': [layer.weights for layer in network.layers],
        'biases': [layer.bias for layer in network.layers],
        'preprocessing': {
            'method': preprocessor.normalization,
            'feature_means': preprocessor.feature_means,
            'feature_stds': preprocessor.feature_stds,
            'feature_mins': preprocessor.feature_mins,
            'feature_maxs': preprocessor.feature_maxs
        },
        'config': {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs_trained': epoch,
            'loss_function': network.loss_name
        }
    }
    
    # Guardar en formato JSON (más legible y portable que pickle)
    try:
        with open(args.save, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"   ✅ Modelo guardado en '{args.save}'")
    except Exception as e:
        print(f"   ❌ Error al guardar modelo: {e}", file=sys.stderr)
    
    # Guardar historial
    try:
        with open(args.history, 'w') as f:
            json.dump(network.history, f, indent=2)
        print(f"   ✅ Historial guardado en '{args.history}'")
    except Exception as e:
        print(f"   ⚠️  No se pudo guardar historial: {e}", file=sys.stderr)
    
    print()
    
    # ========================================================================
    # 5. RESUMEN FINAL
    # ========================================================================
    
    print("=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print()
    
    print("📊 Resultados finales:")
    if has_validation:
        final_train_loss = network.history['loss'][-1]
        final_val_loss = network.history['val_loss'][-1]
        final_train_acc = network.history['accuracy'][-1]
        final_val_acc = network.history['val_accuracy'][-1]
        
        print(f"   Train Loss:      {final_train_loss:.6f}")
        print(f"   Train Accuracy:  {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"   Val Loss:        {final_val_loss:.6f}")
        print(f"   Val Accuracy:    {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    else:
        final_train_loss = network.history['loss'][-1]
        final_train_acc = network.history['accuracy'][-1]
        
        print(f"   Train Loss:      {final_train_loss:.6f}")
        print(f"   Train Accuracy:  {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    
    print()
    print(f"⏱️  Tiempo total: {timer.elapsed_formatted()}")
    print()
    
    # Indicar siguiente paso
    print("📝 Siguiente paso:")
    print(f"   python3 predict.py --dataset {args.validation} --model {args.save}")
    print()
    
    # Generar curvas de aprendizaje
    print("📊 Para visualizar curvas de aprendizaje:")
    print(f"   python3 visualize.py --history {args.history}")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print()
        print("⚠️  Entrenamiento interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print()
        print(f"❌ Error durante el entrenamiento: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
