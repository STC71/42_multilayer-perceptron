#!/usr/bin/env python3
"""
explore_dataset.py - Script de exploración del dataset de cáncer de mama

Analiza el dataset mostrando:
    - Estadísticas generales
    - Distribución de clases
    - Estadísticas de features
    - Detección de valores ausentes
    - Correlaciones básicas

Usage:
    python3 explore_dataset.py data.csv
"""

import sys
import argparse


def load_dataset(filepath):
    """Carga el dataset desde un archivo CSV."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return lines
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{filepath}'", file=sys.stderr)
        sys.exit(1)


def parse_data(lines):
    """
    Parsea las líneas del dataset.
    
    Returns:
        Tupla (ids, labels, features_matrix)
    """
    ids = []
    labels = []
    features = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        values = line.split(',')
        if len(values) != 32:
            continue
        
        ids.append(values[0])
        labels.append(values[1])
        
        try:
            feature_row = [float(val) for val in values[2:]]
            features.append(feature_row)
        except ValueError:
            continue
    
    return ids, labels, features


def calculate_statistics(data):
    """
    Calcula estadísticas básicas para una lista de valores.
    
    Returns:
        Dict con min, max, mean, std
    """
    n = len(data)
    if n == 0:
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
    
    min_val = min(data)
    max_val = max(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std = variance ** 0.5
    
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean,
        'std': std
    }


def explore_dataset(filepath):
    """
    Función principal de exploración del dataset.
    """
    print("=" * 80)
    print("🔬 EXPLORACIÓN DEL DATASET DE CÁNCER DE MAMA")
    print("=" * 80)
    print()
    
    # Cargar datos
    print(f"📂 Cargando dataset desde '{filepath}'...")
    lines = load_dataset(filepath)
    ids, labels, features = parse_data(lines)
    
    n_samples = len(ids)
    n_features = len(features[0]) if features else 0
    
    print(f"   ✓ Muestras cargadas: {n_samples}")
    print(f"   ✓ Features por muestra: {n_features}")
    print()
    
    # Distribución de clases
    print("📊 DISTRIBUCIÓN DE CLASES:")
    print("-" * 80)
    count_m = sum(1 for label in labels if label == 'M')
    count_b = sum(1 for label in labels if label == 'B')
    
    print(f"   Malignos (M):  {count_m:3d} muestras ({count_m/n_samples*100:5.1f}%)")
    print(f"   Benignos (B):  {count_b:3d} muestras ({count_b/n_samples*100:5.1f}%)")
    print(f"   Balance:       {min(count_m, count_b) / max(count_m, count_b) * 100:.1f}%")
    print()
    
    # Estadísticas de features
    print("📈 ESTADÍSTICAS DE FEATURES:")
    print("-" * 80)
    
    # Nombres de features (según el dataset de Wisconsin)
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
        "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ]
    
    print(f"{'Feature':<25} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 80)
    
    for feature_idx in range(min(10, n_features)):  # Mostrar solo las primeras 10
        feature_values = [sample[feature_idx] for sample in features]
        stats = calculate_statistics(feature_values)
        
        name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
        
        print(f"{name:<25} {stats['min']:12.4f} {stats['max']:12.4f} {stats['mean']:12.4f} {stats['std']:12.4f}")
    
    if n_features > 10:
        print(f"   ... ({n_features - 10} features adicionales)")
    print()
    
    # Verificar valores ausentes o anómalos
    print("🔍 VALIDACIÓN DE DATOS:")
    print("-" * 80)
    
    has_issues = False
    
    # Verificar valores ausentes
    for sample_idx, sample in enumerate(features):
        for feature_idx, value in enumerate(sample):
            if value is None or value != value:  # None o NaN
                print(f"   ⚠️  Valor ausente en muestra {sample_idx}, feature {feature_idx}")
                has_issues = True
    
    if not has_issues:
        print("   ✅ No se detectaron valores ausentes")
    print()
    
    # Rangos de features
    print("📏 RANGOS DE FEATURES:")
    print("-" * 80)
    print("   Análisis de escalas para determinar necesidad de normalización:")
    print()
    
    ranges = []
    for feature_idx in range(n_features):
        feature_values = [sample[feature_idx] for sample in features]
        min_val = min(feature_values)
        max_val = max(feature_values)
        ranges.append(max_val - min_val)
    
    min_range = min(ranges)
    max_range = max(ranges)
    
    print(f"   Rango mínimo:  {min_range:.6f}")
    print(f"   Rango máximo:  {max_range:.2f}")
    print(f"   Ratio:         {max_range/min_range:.2f}x")
    print()
    
    if max_range / min_range > 100:
        print("   ⚠️  IMPORTANTE: Las features tienen escalas muy diferentes")
        print("   ➜ Se recomienda NORMALIZACIÓN antes del entrenamiento")
    else:
        print("   ℹ️  Las features tienen escalas relativamente similares")
    print()
    
    # Recomendaciones
    print("💡 RECOMENDACIONES:")
    print("-" * 80)
    print("   1. Normalizar features (Standardization recomendado)")
    print("   2. Dividir dataset 80% train / 20% validation")
    print("   3. Codificar labels: M→1, B→0")
    print("   4. Usar binary cross-entropy como función de pérdida")
    print("   5. Considerar arquitectura: Input(30) → Hidden(24) → Hidden(24) → Output(2)")
    print()
    
    print("=" * 80)
    print("✅ EXPLORACIÓN COMPLETADA")
    print("=" * 80)
    print()
    print("📝 Siguiente paso: python3 split_dataset.py data.csv")
    print()


def main():
    """Punto de entrada del script."""
    parser = argparse.ArgumentParser(
        description='Explora y analiza el dataset de cáncer de mama',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'dataset',
        type=str,
        help='Ruta al archivo CSV del dataset'
    )
    
    args = parser.parse_args()
    explore_dataset(args.dataset)


if __name__ == '__main__':
    main()
