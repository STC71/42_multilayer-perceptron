#!/usr/bin/env python3
"""
initialization.py - Inicialización de pesos para redes neuronales

Implementa métodos de inicialización de pesos óptimos para entrenar redes profundas.
Una buena inicialización es crítica para la convergencia del modelo.

Métodos disponibles:
    - Random: Inicialización aleatoria simple
    - Xavier/Glorot: Óptimo para sigmoid/tanh
    - He: Óptimo para ReLU/Leaky ReLU
    
Referencias:
    - Xavier: "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)
    - He: "Delving Deep into Rectifiers" (He et al., 2015)
"""

import random
import math


def set_random_seed(seed):
    """
    Establece la semilla aleatoria para reproducibilidad.
    
    Args:
        seed: Valor de la semilla
    """
    random.seed(seed)


def random_normal(mean=0.0, std=1.0):
    """
    Genera un número aleatorio con distribución normal.
    
    Usa el método de Box-Muller para generar valores gaussianos.
    
    Args:
        mean: Media de la distribución
        std: Desviación estándar
        
    Returns:
        Número aleatorio con distribución N(mean, std²)
    """
    # Box-Muller transform
    u1 = random.random()
    u2 = random.random()
    
    # Prevenir log(0)
    u1 = max(1e-10, u1)
    
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    
    return mean + z0 * std


def random_uniform(low=0.0, high=1.0):
    """
    Genera un número aleatorio con distribución uniforme.
    
    Args:
        low: Límite inferior
        high: Límite superior
        
    Returns:
        Número aleatorio en [low, high)
    """
    return low + (high - low) * random.random()


# ============================================================================
# INICIALIZACIÓN RANDOM (Baseline)
# ============================================================================

def initialize_random(n_in, n_out, scale=0.01):
    """
    Inicialización aleatoria simple.
    
    Genera pesos pequeños aleatorios centrados en cero.
    No es óptimo, pero sirve como baseline.
    
    Args:
        n_in: Número de neuronas en la capa anterior
        n_out: Número de neuronas en la capa actual
        scale: Factor de escala (típicamente 0.01)
        
    Returns:
        Matriz de pesos (n_out × n_in)
    """
    weights = []
    for i in range(n_out):
        row = []
        for j in range(n_in):
            # Distribución uniforme pequeña
            weight = random_uniform(-scale, scale)
            row.append(weight)
        weights.append(row)
    
    return weights


# ============================================================================
# XAVIER / GLOROT INITIALIZATION
# ============================================================================

def initialize_xavier_normal(n_in, n_out):
    """
    Inicialización Xavier/Glorot con distribución normal.
    
    Óptimo para: sigmoid, tanh
    
    Formula: W ~ N(0, √(1/n_in))
    
    Mantiene la varianza de las activaciones constante a través de las capas.
    
    Args:
        n_in: Número de neuronas en la capa anterior
        n_out: Número de neuronas en la capa actual
        
    Returns:
        Matriz de pesos (n_out × n_in)
    """
    std = math.sqrt(1.0 / n_in)
    
    weights = []
    for i in range(n_out):
        row = []
        for j in range(n_in):
            weight = random_normal(mean=0.0, std=std)
            row.append(weight)
        weights.append(row)
    
    return weights


def initialize_xavier_uniform(n_in, n_out):
    """
    Inicialización Xavier/Glorot con distribución uniforme.
    
    Óptimo para: sigmoid, tanh
    
    Formula: W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
    
    Variante uniforme que también mantiene la varianza constante.
    
    Args:
        n_in: Número de neuronas en la capa anterior
        n_out: Número de neuronas en la capa actual
        
    Returns:
        Matriz de pesos (n_out × n_in)
    """
    limit = math.sqrt(6.0 / (n_in + n_out))
    
    weights = []
    for i in range(n_out):
        row = []
        for j in range(n_in):
            weight = random_uniform(-limit, limit)
            row.append(weight)
        weights.append(row)
    
    return weights


# ============================================================================
# HE INITIALIZATION
# ============================================================================

def initialize_he_normal(n_in, n_out):
    """
    Inicialización He con distribución normal.
    
    Óptimo para: ReLU, Leaky ReLU
    
    Formula: W ~ N(0, √(2/n_in))
    
    Diseñada específicamente para funciones de activación ReLU,
    que "matan" la mitad de las neuronas (valores negativos → 0).
    
    Args:
        n_in: Número de neuronas en la capa anterior
        n_out: Número de neuronas en la capa actual
        
    Returns:
        Matriz de pesos (n_out × n_in)
    """
    std = math.sqrt(2.0 / n_in)
    
    weights = []
    for i in range(n_out):
        row = []
        for j in range(n_in):
            weight = random_normal(mean=0.0, std=std)
            row.append(weight)
        weights.append(row)
    
    return weights


def initialize_he_uniform(n_in, n_out):
    """
    Inicialización He con distribución uniforme.
    
    Óptimo para: ReLU, Leaky ReLU
    
    Formula: W ~ U(-√(6/n_in), √(6/n_in))
    
    Variante uniforme de He initialization.
    
    Args:
        n_in: Número de neuronas en la capa anterior
        n_out: Número de neuronas en la capa actual
        
    Returns:
        Matriz de pesos (n_out × n_in)
    """
    limit = math.sqrt(6.0 / n_in)
    
    weights = []
    for i in range(n_out):
        row = []
        for j in range(n_in):
            weight = random_uniform(-limit, limit)
            row.append(weight)
        weights.append(row)
    
    return weights


# ============================================================================
# INICIALIZACIÓN DE BIAS
# ============================================================================

def initialize_bias_zeros(n):
    """
    Inicializa bias a cero.
    
    Práctica estándar: los bias suelen inicializarse a 0.
    
    Args:
        n: Número de neuronas (tamaño del vector de bias)
        
    Returns:
        Vector de bias (lista de n ceros)
    """
    return [0.0] * n


def initialize_bias_small(n, value=0.01):
    """
    Inicializa bias a un valor pequeño constante.
    
    Alternativa para ReLU: evita "dead neurons" al inicio.
    
    Args:
        n: Número de neuronas
        value: Valor constante pequeño (típicamente 0.01)
        
    Returns:
        Vector de bias
    """
    return [value] * n


# ============================================================================
# DISPATCHER PRINCIPAL
# ============================================================================

def initialize_weights(n_in, n_out, method='he', activation='relu', seed=None):
    """
    Inicializa una matriz de pesos según el método especificado.
    
    Selecciona automáticamente el mejor método según la activación si no se especifica.
    
    Args:
        n_in: Número de neuronas en la capa anterior
        n_out: Número de neuronas en la capa actual
        method: Método de inicialización ('random', 'xavier', 'he', 'xavier_uniform', 'he_uniform')
        activation: Función de activación (para auto-selección)
        seed: Semilla aleatoria (opcional)
        
    Returns:
        Matriz de pesos (n_out × n_in)
    """
    if seed is not None:
        set_random_seed(seed)
    
    # Auto-selección del método según la activación
    if method == 'auto':
        if activation in ['relu', 'leaky_relu']:
            method = 'he'
        elif activation in ['sigmoid', 'tanh']:
            method = 'xavier'
        else:
            method = 'xavier'
    
    # Mapeo de métodos
    initializers = {
        'random': lambda: initialize_random(n_in, n_out),
        'xavier': lambda: initialize_xavier_normal(n_in, n_out),
        'xavier_normal': lambda: initialize_xavier_normal(n_in, n_out),
        'xavier_uniform': lambda: initialize_xavier_uniform(n_in, n_out),
        'glorot': lambda: initialize_xavier_normal(n_in, n_out),
        'he': lambda: initialize_he_normal(n_in, n_out),
        'he_normal': lambda: initialize_he_normal(n_in, n_out),
        'he_uniform': lambda: initialize_he_uniform(n_in, n_out)
    }
    
    method_lower = method.lower()
    if method_lower not in initializers:
        raise ValueError(f"Método de inicialización desconocido: {method}")
    
    return initializers[method_lower]()


def initialize_bias(n, method='zeros'):
    """
    Inicializa un vector de bias.
    
    Args:
        n: Número de neuronas
        method: Método de inicialización ('zeros' o 'small')
        
    Returns:
        Vector de bias (lista de n valores)
    """
    if method == 'zeros':
        return initialize_bias_zeros(n)
    elif method == 'small':
        return initialize_bias_small(n)
    else:
        raise ValueError(f"Método de inicialización de bias desconocido: {method}")


# ============================================================================
# TESTS
# ============================================================================

def test_initializations():
    """
    Tests básicos para validar las inicializaciones.
    """
    print("🧪 Testeando inicialización de pesos...")
    print()
    
    n_in, n_out = 10, 5
    
    # Test Xavier
    print("1️⃣  Xavier Normal:")
    W_xavier = initialize_xavier_normal(n_in, n_out)
    flat = [w for row in W_xavier for w in row]
    mean = sum(flat) / len(flat)
    variance = sum((w - mean)**2 for w in flat) / len(flat)
    print(f"   Shape: {n_out} × {n_in}")
    print(f"   Media: {mean:.6f} (esperado: ~0.0)")
    print(f"   Varianza: {variance:.6f} (esperado: ~{1.0/n_in:.6f})")
    print()
    
    # Test He
    print("2️⃣  He Normal:")
    W_he = initialize_he_normal(n_in, n_out)
    flat = [w for row in W_he for w in row]
    mean = sum(flat) / len(flat)
    variance = sum((w - mean)**2 for w in flat) / len(flat)
    print(f"   Shape: {n_out} × {n_in}")
    print(f"   Media: {mean:.6f} (esperado: ~0.0)")
    print(f"   Varianza: {variance:.6f} (esperado: ~{2.0/n_in:.6f})")
    print()
    
    # Test Bias
    print("3️⃣  Bias:")
    b = initialize_bias(5, 'zeros')
    print(f"   Zeros: {b}")
    b_small = initialize_bias(5, 'small')
    print(f"   Small: {b_small}")
    print()
    
    print("✅ Tests completados")
    print()


if __name__ == '__main__':
    test_initializations()
