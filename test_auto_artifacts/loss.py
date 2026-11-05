#!/usr/bin/env python3
"""
loss.py - Funciones de pérdida para redes neuronales

Implementa funciones de pérdida (loss functions) y sus derivadas para backpropagation.

REQUERIDO POR EL SUBJECT:
    - Binary Cross-Entropy: E = -1/N × Σ[yn log(pn) + (1-yn) log(1-pn)]

Funciones disponibles:
    - Binary Cross-Entropy: Para clasificación binaria
    - Categorical Cross-Entropy: Para clasificación multiclase
    - Mean Squared Error: Para regresión (bonus)
    
Referencias:
    - Subject multilayer-perceptron.pdf, página 10
"""

import math
import sys


# ============================================================================
# BINARY CROSS-ENTROPY (Obligatorio por el subject)
# ============================================================================

def binary_crossentropy(y_true, y_pred, epsilon=1e-10):
    """
    Función de pérdida Binary Cross-Entropy.
    
    FÓRMULA EXACTA DEL SUBJECT:
    E = -1/N × Σ[yn log(pn) + (1-yn) log(1-pn)]
    
    Donde:
        - N = número de muestras
        - yn = label verdadero (0 o 1)
        - pn = probabilidad predicha
    
    Args:
        y_true: Labels verdaderos (lista de 0s y 1s)
        y_pred: Probabilidades predichas (lista de valores en [0, 1])
        epsilon: Pequeño valor para evitar log(0)
        
    Returns:
        Valor de pérdida (float)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener el mismo tamaño")
    
    N = len(y_true)
    if N == 0:
        return 0.0
    
    total_loss = 0.0
    
    for yn, pn in zip(y_true, y_pred):
        # Clip para evitar log(0)
        pn = max(epsilon, min(1.0 - epsilon, pn))
        
        # Aplicar la fórmula del subject
        loss_n = yn * math.log(pn) + (1 - yn) * math.log(1 - pn)
        total_loss += loss_n
    
    # Negativo y promedio
    return -total_loss / N


def binary_crossentropy_derivative(y_true, y_pred, epsilon=1e-10):
    """
    Derivada de Binary Cross-Entropy respecto a las predicciones.
    
    Formula: ∂E/∂pn = -[yn/pn - (1-yn)/(1-pn)]
    
    Args:
        y_true: Labels verdaderos
        y_pred: Probabilidades predichas
        epsilon: Pequeño valor para prevenir división por cero
        
    Returns:
        Lista de derivadas
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener el mismo tamaño")
    
    N = len(y_true)
    derivatives = []
    
    for yn, pn in zip(y_true, y_pred):
        # Clip para evitar división por cero
        pn = max(epsilon, min(1.0 - epsilon, pn))
        
        # Derivada
        deriv = -(yn / pn - (1 - yn) / (1 - pn))
        derivatives.append(deriv / N)  # Dividir por N para promediar
    
    return derivatives


# ============================================================================
# CATEGORICAL CROSS-ENTROPY (Para softmax multiclase)
# ============================================================================

def categorical_crossentropy(y_true, y_pred, epsilon=1e-10):
    """
    Función de pérdida Categorical Cross-Entropy.
    
    Formula: E = -Σ Σ yij log(pij)
             i  j
    
    Donde:
        - i = índice de muestra
        - j = índice de clase
        - yij = label verdadero (one-hot encoded)
        - pij = probabilidad predicha para clase j
    
    Uso: Clasificación multiclase con softmax
    
    Args:
        y_true: Labels verdaderos (lista de listas, one-hot encoded)
                Ejemplo: [[1, 0], [0, 1], [1, 0]] para 3 muestras, 2 clases
        y_pred: Probabilidades predichas (lista de listas, salida de softmax)
        epsilon: Pequeño valor para evitar log(0)
        
    Returns:
        Valor de pérdida (float)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener el mismo tamaño")
    
    N = len(y_true)
    if N == 0:
        return 0.0
    
    total_loss = 0.0
    
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        if len(y_true_sample) != len(y_pred_sample):
            raise ValueError("Las dimensiones de las clases no coinciden")
        
        for yj, pj in zip(y_true_sample, y_pred_sample):
            # Clip para evitar log(0)
            pj = max(epsilon, min(1.0 - epsilon, pj))
            
            # Solo sumar si la clase verdadera es 1
            if yj == 1:
                total_loss += math.log(pj)
    
    return -total_loss / N


def categorical_crossentropy_derivative(y_true, y_pred, epsilon=1e-10):
    """
    Derivada de Categorical Cross-Entropy.
    
    Cuando se usa con Softmax, la derivada se simplifica a:
        ∂E/∂z = pred - true
    
    Args:
        y_true: Labels verdaderos (one-hot)
        y_pred: Probabilidades predichas
        epsilon: Pequeño valor para estabilidad numérica
        
    Returns:
        Lista de listas con derivadas
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener el mismo tamaño")
    
    N = len(y_true)
    derivatives = []
    
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        deriv_sample = []
        for yj, pj in zip(y_true_sample, y_pred_sample):
            # Clip para prevenir división por cero
            pj = max(epsilon, min(1.0 - epsilon, pj))
            
            # Derivada: -y/p
            deriv = -yj / pj
            deriv_sample.append(deriv / N)
        
        derivatives.append(deriv_sample)
    
    return derivatives


# ============================================================================
# MEAN SQUARED ERROR (Bonus - para regresión)
# ============================================================================

def mean_squared_error(y_true, y_pred):
    """
    Función de pérdida Mean Squared Error (MSE).
    
    Formula: MSE = 1/N × Σ(yn - pn)²
    
    Uso: Problemas de regresión
    
    Args:
        y_true: Valores verdaderos
        y_pred: Valores predichos
        
    Returns:
        Valor de pérdida (float)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener el mismo tamaño")
    
    N = len(y_true)
    if N == 0:
        return 0.0
    
    total_loss = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred))
    return total_loss / N


def mean_squared_error_derivative(y_true, y_pred):
    """
    Derivada de Mean Squared Error.
    
    Formula: ∂MSE/∂pn = 2/N × (pn - yn)
    
    Args:
        y_true: Valores verdaderos
        y_pred: Valores predichos
        
    Returns:
        Lista de derivadas
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener el mismo tamaño")
    
    N = len(y_true)
    return [2.0 * (yp - yt) / N for yt, yp in zip(y_true, y_pred)]


# ============================================================================
# DISPATCHER
# ============================================================================

LOSS_FUNCTIONS = {
    'binary_crossentropy': binary_crossentropy,
    'categorical_crossentropy': categorical_crossentropy,
    'mse': mean_squared_error,
    'mean_squared_error': mean_squared_error
}

LOSS_DERIVATIVES = {
    'binary_crossentropy': binary_crossentropy_derivative,
    'categorical_crossentropy': categorical_crossentropy_derivative,
    'mse': mean_squared_error_derivative,
    'mean_squared_error': mean_squared_error_derivative
}


def get_loss_function(name):
    """
    Obtiene una función de pérdida por nombre.
    
    Args:
        name: Nombre de la función
        
    Returns:
        Función de pérdida
    """
    name = name.lower()
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Función de pérdida desconocida: {name}")
    return LOSS_FUNCTIONS[name]


def get_loss_derivative(name):
    """
    Obtiene la derivada de una función de pérdida por nombre.
    
    Args:
        name: Nombre de la función
        
    Returns:
        Función derivada
    """
    name = name.lower()
    if name not in LOSS_DERIVATIVES:
        raise ValueError(f"Derivada de pérdida desconocida: {name}")
    return LOSS_DERIVATIVES[name]


# ============================================================================
# TESTS
# ============================================================================

def test_loss_functions():
    """
    Tests básicos para validar las funciones de pérdida.
    """
    print("🧪 Testeando funciones de pérdida...")
    print()
    
    # Test Binary Cross-Entropy
    print("1️⃣  Binary Cross-Entropy:")
    y_true = [1, 0, 1, 0]
    y_pred = [0.9, 0.1, 0.8, 0.2]
    loss = binary_crossentropy(y_true, y_pred)
    print(f"   Loss: {loss:.6f}")
    print(f"   y_true: {y_true}")
    print(f"   y_pred: {y_pred}")
    
    # Test con predicción perfecta
    y_pred_perfect = [1.0, 0.0, 1.0, 0.0]
    loss_perfect = binary_crossentropy(y_true, y_pred_perfect)
    print(f"   Loss (perfect): {loss_perfect:.6f} (esperado: ~0.0)")
    
    # Test con predicción aleatoria
    y_pred_random = [0.5, 0.5, 0.5, 0.5]
    loss_random = binary_crossentropy(y_true, y_pred_random)
    print(f"   Loss (random): {loss_random:.6f} (esperado: ~0.693)")
    print()
    
    # Test Categorical Cross-Entropy
    print("2️⃣  Categorical Cross-Entropy:")
    y_true_cat = [[1, 0], [0, 1], [1, 0]]
    y_pred_cat = [[0.9, 0.1], [0.2, 0.8], [0.85, 0.15]]
    loss_cat = categorical_crossentropy(y_true_cat, y_pred_cat)
    print(f"   Loss: {loss_cat:.6f}")
    print()
    
    # Test MSE
    print("3️⃣  Mean Squared Error:")
    y_true_mse = [1.0, 2.0, 3.0, 4.0]
    y_pred_mse = [1.1, 1.9, 3.2, 3.8]
    loss_mse = mean_squared_error(y_true_mse, y_pred_mse)
    print(f"   MSE: {loss_mse:.6f}")
    print()
    
    print("✅ Todos los tests completados")
    print()


if __name__ == '__main__':
    test_loss_functions()
