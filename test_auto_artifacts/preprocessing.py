#!/usr/bin/env python3
"""
preprocessing.py - Módulo de preprocesamiento de datos

Función: Normalizar features y codificar etiquetas para el entrenamiento de la red neuronal.

Incluye:
    - Normalización Min-Max
    - Normalización Z-Score (Standardization)
    - Codificación de etiquetas (M/B -> 0/1)
    - Manejo de valores ausentes
"""

import sys


class DataPreprocessor:
    """
    Clase para preprocesar datos del dataset de cáncer de mama.
    
    Maneja:
    - Normalización de features
    - Codificación de etiquetas
    - División de datos en X (features) e y (labels)
    """
    
    def __init__(self, normalization='standardization'):
        """
        Inicializa el preprocesador.
        
        Args:
            normalization: Tipo de normalización ('min-max' o 'standardization')
        """
        self.normalization = normalization
        self.feature_mins = None
        self.feature_maxs = None
        self.feature_means = None
        self.feature_stds = None
        self.fitted = False
        
    def load_data(self, filepath):
        """
        Carga datos desde un archivo CSV sin header.
        
        Format: ID,Diagnosis,Feature1,Feature2,...,Feature30
        
        Args:
            filepath: Ruta al archivo CSV
            
        Returns:
            Tupla (features, labels, ids)
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo '{filepath}'", file=sys.stderr)
            sys.exit(1)
        
        ids = []
        labels = []
        features = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            values = line.split(',')
            
            # Verificar que la línea tenga el formato correcto (32 columnas)
            if len(values) != 32:
                print(f"⚠️  Advertencia: Línea con formato incorrecto ignorada (esperadas 32 columnas, encontradas {len(values)})", file=sys.stderr)
                continue
            
            # Extraer componentes
            ids.append(values[0])
            labels.append(values[1])
            
            # Convertir features a float
            try:
                feature_row = [float(val) for val in values[2:]]
                features.append(feature_row)
            except ValueError as e:
                print(f"⚠️  Advertencia: Error al convertir valores a float: {e}", file=sys.stderr)
                continue
        
        return features, labels, ids
    
    def encode_labels(self, labels):
        """
        Codifica las etiquetas M/B a formato numérico.
        
        M (Maligno) -> 1
        B (Benigno) -> 0
        
        Args:
            labels: Lista de etiquetas ['M', 'B', ...]
            
        Returns:
            Lista de etiquetas numéricas [1, 0, ...]
        """
        encoded = []
        for label in labels:
            if label == 'M':
                encoded.append(1)
            elif label == 'B':
                encoded.append(0)
            else:
                print(f"❌ Error: Etiqueta desconocida '{label}' (esperado 'M' o 'B')", file=sys.stderr)
                sys.exit(1)
        return encoded
    
    def fit(self, features):
        """
        Calcula estadísticas necesarias para la normalización (debe usarse solo con train data).
        
        Args:
            features: Lista de listas con features [[f1, f2, ...], ...]
        """
        n_samples = len(features)
        n_features = len(features[0])
        
        # Inicializar estadísticas
        self.feature_mins = [float('inf')] * n_features
        self.feature_maxs = [float('-inf')] * n_features
        self.feature_means = [0.0] * n_features
        self.feature_stds = [0.0] * n_features
        
        # Calcular mínimos, máximos y medias
        for feature_idx in range(n_features):
            feature_values = [sample[feature_idx] for sample in features]
            
            self.feature_mins[feature_idx] = min(feature_values)
            self.feature_maxs[feature_idx] = max(feature_values)
            self.feature_means[feature_idx] = sum(feature_values) / n_samples
        
        # Calcular desviaciones estándar
        for feature_idx in range(n_features):
            feature_values = [sample[feature_idx] for sample in features]
            variance = sum((val - self.feature_means[feature_idx]) ** 2 for val in feature_values) / n_samples
            self.feature_stds[feature_idx] = variance ** 0.5
            
            # Evitar división por cero
            if self.feature_stds[feature_idx] == 0:
                self.feature_stds[feature_idx] = 1.0
        
        self.fitted = True
        
    def transform(self, features):
        """
        Normaliza las features usando estadísticas previamente calculadas.
        
        Args:
            features: Lista de listas con features
            
        Returns:
            Features normalizadas
        """
        if not self.fitted:
            raise ValueError("❌ Error: Debe llamar a fit() antes de transform()")
        
        normalized = []
        
        for sample in features:
            normalized_sample = []
            
            for feature_idx, value in enumerate(sample):
                if self.normalization == 'min-max':
                    # Normalización Min-Max: (x - min) / (max - min)
                    min_val = self.feature_mins[feature_idx]
                    max_val = self.feature_maxs[feature_idx]
                    
                    if max_val - min_val == 0:
                        normalized_value = 0.0
                    else:
                        normalized_value = (value - min_val) / (max_val - min_val)
                
                elif self.normalization == 'standardization':
                    # Normalización Z-Score: (x - mean) / std
                    mean = self.feature_means[feature_idx]
                    std = self.feature_stds[feature_idx]
                    normalized_value = (value - mean) / std
                
                else:
                    normalized_value = value
                
                normalized_sample.append(normalized_value)
            
            normalized.append(normalized_sample)
        
        return normalized
    
    def fit_transform(self, features):
        """
        Calcula estadísticas y normaliza en un solo paso.
        
        Args:
            features: Lista de listas con features
            
        Returns:
            Features normalizadas
        """
        self.fit(features)
        return self.transform(features)


def split_data(filepath, train_ratio=0.8, seed=42):
    """
    Función principal que divide el dataset.
    
    Args:
        filepath: Ruta al archivo CSV
        train_ratio: Proporción para entrenamiento
        seed: Semilla aleatoria
    """
    print("=" * 70)
    print("🔀 DIVISIÓN DEL DATASET DE CÁNCER DE MAMA")
    print("=" * 70)
    print()
    
    # Cargar dataset
    print(f"📂 Cargando dataset desde '{filepath}'...")
    data = load_dataset(filepath)
    total_samples = len(data)
    print(f"   ✓ Total de muestras cargadas: {total_samples}")
    print()
    
    # Configuración
    print("⚙️  Configuración:")
    print(f"   • Ratio entrenamiento: {train_ratio:.0%}")
    print(f"   • Ratio validación: {1-train_ratio:.0%}")
    print(f"   • Semilla aleatoria: {seed}")
    print()
    
    # Dividir
    print("✂️  Dividiendo dataset...")
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    split_index = int(total_samples * train_ratio)
    train_data = shuffled_data[:split_index]
    valid_data = shuffled_data[split_index:]
    
    train_size = len(train_data)
    valid_size = len(valid_data)
    
    print(f"   ✓ Entrenamiento: {train_size} muestras ({train_size/total_samples*100:.1f}%)")
    print(f"   ✓ Validación: {valid_size} muestras ({valid_size/total_samples*100:.1f}%)")
    print()
    
    # Guardar
    print("💾 Guardando datasets divididos...")
    save_dataset('data_training.csv', train_data)
    save_dataset('data_validation.csv', valid_data)
    print()
    
    # Análisis de distribución de clases
    print("📊 Distribución de clases:")
    train_m = sum(1 for line in train_data if ',M,' in line)
    train_b = train_size - train_m
    valid_m = sum(1 for line in valid_data if ',M,' in line)
    valid_b = valid_size - valid_m
    
    print(f"   Entrenamiento:")
    print(f"      • Malignos (M): {train_m} ({train_m/train_size*100:.1f}%)")
    print(f"      • Benignos (B): {train_b} ({train_b/train_size*100:.1f}%)")
    print(f"   Validación:")
    print(f"      • Malignos (M): {valid_m} ({valid_m/valid_size*100:.1f}%)")
    print(f"      • Benignos (B): {valid_b} ({valid_b/valid_size*100:.1f}%)")
    print()
    
    print("=" * 70)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 70)


if __name__ == '__main__':
    args = parse_arguments()
    split_data(args.dataset, args.train_ratio, args.seed)
