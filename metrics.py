#!/usr/bin/env python3
"""
metrics.py - Métricas de evaluación para clasificación

Implementa métricas para evaluar el rendimiento del modelo:
    - Accuracy (Exactitud)
    - Precision (Precisión)
    - Recall (Sensibilidad)
    - F1-Score
    - Confusion Matrix

REQUERIDO POR EL SUBJECT:
    - Accuracy para mostrar en las curvas de aprendizaje

BONUS:
    - Métricas adicionales para evaluación completa
"""


def compute_accuracy(y_true, y_pred):
    """
    Calcula la accuracy (exactitud).
    
    Formula: Accuracy = (TP + TN) / Total
    
    Args:
        y_true: Labels verdaderos (lista de 0s y 1s)
        y_pred: Predicciones (lista de probabilidades o one-hot)
        
    Returns:
        Accuracy (float entre 0 y 1)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener el mismo tamaño")
    
    correct = 0
    total = len(y_true)
    
    for yt, yp in zip(y_true, y_pred):
        # Convertir predicción a clase
        if isinstance(yp, list):
            # Output one-hot: tomar argmax
            pred_class = yp.index(max(yp))
        else:
            # Single value: umbral 0.5
            pred_class = 1 if yp >= 0.5 else 0
        
        # Comparar
        if pred_class == int(yt):
            correct += 1
    
    return correct / total if total > 0 else 0.0


def compute_confusion_matrix(y_true, y_pred, n_classes=2):
    """
    Calcula la matriz de confusión.
    
    Para clasificación binaria:
        [[TN, FP],
         [FN, TP]]
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        n_classes: Número de clases
        
    Returns:
        Matriz de confusión (lista de listas)
    """
    # Inicializar matriz
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    
    for yt, yp in zip(y_true, y_pred):
        # Convertir predicción a clase
        if isinstance(yp, list):
            pred_class = yp.index(max(yp))
        else:
            pred_class = 1 if yp >= 0.5 else 0
        
        true_class = int(yt)
        matrix[true_class][pred_class] += 1
    
    return matrix


def get_binary_metrics(confusion_matrix):
    """
    Extrae métricas binarias de la matriz de confusión.
    
    Args:
        confusion_matrix: Matriz 2×2 [[TN, FP], [FN, TP]]
        
    Returns:
        Dict con TP, TN, FP, FN
    """
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    
    return {
        'TP': TP,  # True Positives
        'TN': TN,  # True Negatives
        'FP': FP,  # False Positives
        'FN': FN   # False Negatives
    }


def compute_precision(y_true, y_pred):
    """
    Calcula la precisión (precision).
    
    Formula: Precision = TP / (TP + FP)
    
    Interpretación: De todos los que predijimos como positivos,
    ¿cuántos realmente lo son?
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        
    Returns:
        Precision (float entre 0 y 1)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    metrics = get_binary_metrics(cm)
    
    TP = metrics['TP']
    FP = metrics['FP']
    
    if TP + FP == 0:
        return 0.0
    
    return TP / (TP + FP)


def compute_recall(y_true, y_pred):
    """
    Calcula el recall (sensibilidad, sensitivity).
    
    Formula: Recall = TP / (TP + FN)
    
    Interpretación: De todos los positivos reales,
    ¿cuántos detectamos correctamente?
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        
    Returns:
        Recall (float entre 0 y 1)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    metrics = get_binary_metrics(cm)
    
    TP = metrics['TP']
    FN = metrics['FN']
    
    if TP + FN == 0:
        return 0.0
    
    return TP / (TP + FN)


def compute_f1_score(y_true, y_pred):
    """
    Calcula el F1-Score.
    
    Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
    
    Es la media armónica entre precision y recall.
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        
    Returns:
        F1-Score (float entre 0 y 1)
    """
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def compute_specificity(y_true, y_pred):
    """
    Calcula la especificidad.
    
    Formula: Specificity = TN / (TN + FP)
    
    Interpretación: De todos los negativos reales,
    ¿cuántos identificamos correctamente?
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        
    Returns:
        Specificity (float entre 0 y 1)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    metrics = get_binary_metrics(cm)
    
    TN = metrics['TN']
    FP = metrics['FP']
    
    if TN + FP == 0:
        return 0.0
    
    return TN / (TN + FP)


def print_evaluation_report(y_true, y_pred):
    """
    Imprime un reporte completo de evaluación.
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
    """
    print("=" * 80)
    print("📊 REPORTE DE EVALUACIÓN")
    print("=" * 80)
    print()
    
    # Métricas básicas
    accuracy = compute_accuracy(y_true, y_pred)
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    f1 = compute_f1_score(y_true, y_pred)
    specificity = compute_specificity(y_true, y_pred)
    
    print("📈 Métricas:")
    print(f"   Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision:   {precision:.4f}")
    print(f"   Recall:      {recall:.4f}")
    print(f"   F1-Score:    {f1:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print()
    
    # Matriz de confusión
    cm = compute_confusion_matrix(y_true, y_pred)
    metrics = get_binary_metrics(cm)
    
    print("🎯 Matriz de Confusión:")
    print()
    print("                Predicho")
    print("              Benigno  Maligno")
    print(f"    Benigno     {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"    Maligno     {cm[1][0]:4d}     {cm[1][1]:4d}")
    print()
    
    print("📊 Desglose:")
    print(f"   True Positives (TP):  {metrics['TP']:4d}  (Maligno correctamente identificado)")
    print(f"   True Negatives (TN):  {metrics['TN']:4d}  (Benigno correctamente identificado)")
    print(f"   False Positives (FP): {metrics['FP']:4d}  (Benigno predicho como Maligno)")
    print(f"   False Negatives (FN): {metrics['FN']:4d}  (Maligno predicho como Benigno)")
    print()
    
    total = sum(metrics.values())
    print(f"   Total de muestras: {total}")
    print()
    
    print("=" * 80)


def compute_all_metrics(y_true, y_pred):
    """
    Calcula todas las métricas y las retorna en un diccionario.
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        
    Returns:
        Dict con todas las métricas
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    metrics = get_binary_metrics(cm)
    
    return {
        'accuracy': compute_accuracy(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred),
        'recall': compute_recall(y_true, y_pred),
        'f1_score': compute_f1_score(y_true, y_pred),
        'specificity': compute_specificity(y_true, y_pred),
        'confusion_matrix': cm,
        'TP': metrics['TP'],
        'TN': metrics['TN'],
        'FP': metrics['FP'],
        'FN': metrics['FN']
    }


# ============================================================================
# TESTS
# ============================================================================

def test_metrics():
    """
    Tests básicos para validar las métricas.
    """
    print("🧪 Testeando métricas de evaluación...")
    print()
    
    # Dataset de ejemplo
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0.9, 0.1, 0.8, 0.6, 0.2, 0.3, 0.7, 0.4]  # Probabilidades
    
    print("📊 Dataset de ejemplo:")
    print(f"   y_true: {y_true}")
    print(f"   y_pred: {[f'{p:.1f}' for p in y_pred]}")
    print()
    
    # Calcular métricas
    print_evaluation_report(y_true, y_pred)
    
    print("✅ Tests completados")
    print()


if __name__ == '__main__':
    test_metrics()
