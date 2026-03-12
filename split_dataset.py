#!/usr/bin/env python3
"""
split_dataset.py - Divide el dataset en conjuntos de entrenamiento y validación

Usage:
    python3 split_dataset.py data.csv [--train_ratio 0.8] [--seed 42]

Genera:
    - data_training.csv: Dataset de entrenamiento (80% por defecto)
    - data_validation.csv: Dataset de validación (20% por defecto)
"""

import sys
import random
import argparse


def load_dataset(filepath):
    """
    Carga el dataset desde un archivo CSV.
    
    Args:
        filepath: Ruta al archivo CSV
        
    Returns:
        Lista de líneas del dataset
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return lines
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}", file=sys.stderr)
        sys.exit(1)


def save_dataset(filepath, data):
    """
    Guarda el dataset en un archivo CSV.
    
    Args:
        filepath: Ruta donde guardar el archivo
        data: Lista de líneas a guardar
    """
    try:
        with open(filepath, 'w') as f:
            f.writelines(data)
        print(f"✅ Guardado: {filepath} ({len(data)} muestras)")
    except Exception as e:
        print(f"❌ Error al guardar {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def split_dataset(data, train_ratio=0.8, seed=42):
    """
    Divide el dataset en conjuntos de entrenamiento y validación.
    
    Args:
        data: Lista de líneas del dataset
        train_ratio: Proporción de datos para entrenamiento (0.0 - 1.0)
        seed: Semilla para reproducibilidad
        
    Returns:
        Tupla (train_data, validation_data)
    """
    # Establecer semilla para reproducibilidad
    random.seed(seed)
    
    # Mezclar el dataset aleatoriamente
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calcular el punto de división
    split_index = int(len(shuffled_data) * train_ratio)
    
    # Dividir
    train_data = shuffled_data[:split_index]
    validation_data = shuffled_data[split_index:]
    
    return train_data, validation_data


def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        Namespace con los argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Divide el dataset en conjuntos de entrenamiento y validación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 split_dataset.py data.csv
  python3 split_dataset.py data.csv --train_ratio 0.75
  python3 split_dataset.py data.csv --seed 123
        """
    )
    
    parser.add_argument(
        'dataset',
        type=str,
        help='Ruta al archivo CSV del dataset'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Proporción de datos para entrenamiento (default: 0.8)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad (default: 42)'
    )
    
    parser.add_argument(
        '--output_train',
        type=str,
        default='data_training.csv',
        help='Nombre del archivo de salida para entrenamiento (default: data_training.csv)'
    )
    
    parser.add_argument(
        '--output_valid',
        type=str,
        default='data_validation.csv',
        help='Nombre del archivo de salida para validación (default: data_validation.csv)'
    )
    
    return parser.parse_args()


def main():
    """
    Función principal del programa.
    """
    # Parsear argumentos
    args = parse_arguments()
    
    # Validar train_ratio
    if not 0.0 < args.train_ratio < 1.0:
        print("❌ Error: train_ratio debe estar entre 0.0 y 1.0", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60)
    print("🔀 DIVISIÓN DEL DATASET")
    print("=" * 60)
    print(f"📂 Dataset de entrada: {args.dataset}")
    print(f"🎲 Semilla aleatoria: {args.seed}")
    print(f"📊 Ratio entrenamiento/validación: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    print()
    
    # Cargar el dataset
    print("📖 Cargando dataset...")
    data = load_dataset(args.dataset)
    total_samples = len(data)
    print(f"   Total de muestras: {total_samples}")
    print()
    
    # Dividir el dataset
    print("✂️  Dividiendo dataset...")
    train_data, validation_data = split_dataset(
        data, 
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    train_size = len(train_data)
    valid_size = len(validation_data)
    
    print(f"   🎯 Entrenamiento: {train_size} muestras ({train_size/total_samples*100:.1f}%)")
    print(f"   🔍 Validación: {valid_size} muestras ({valid_size/total_samples*100:.1f}%)")
    print()
    
    # Guardar los datasets
    print("💾 Guardando datasets...")
    save_dataset(args.output_train, train_data)
    save_dataset(args.output_valid, validation_data)
    print()
    
    print("=" * 60)
    print("✅ DIVISIÓN COMPLETADA CON ÉXITO")
    print("=" * 60)
    print()
    print("📋 Archivos generados:")
    print(f"   • {args.output_train}")
    print(f"   • {args.output_valid}")
    print()


if __name__ == '__main__':
    main()
