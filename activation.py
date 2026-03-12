#!/usr/bin/env python3
"""
activation.py - Funciones de activación para redes neuronales

Implementa funciones de activación y sus derivadas para backpropagation.
Todas las implementaciones están hechas desde cero sin librerías de ML.

Funciones disponibles:
    - Sigmoid (σ): Para capas ocultas
    - Tanh (tanh): Para capas ocultas
    - ReLU: Para capas ocultas
    - Softmax: Para capa de salida (clasificación multiclase)
    
Cada función incluye:
    - forward(): Calcula la activación
    - derivative(): Calcula la derivada (para backpropagation)
"""

import math


def clip_values(values, min_val=-500, max_val=500):
    """
    Recorta valores para prevenir overflow en exponenciales.
    
    Args:
        values: Lista de valores numéricos
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido
        
    Returns:
        Lista de valores recortados
    """
    return [max(min_val, min(max_val, val)) for val in values]


# ============================================================================
# SIGMOID
# ============================================================================

def sigmoid(x):
    """
    Función de activación Sigmoid.
    
    Formula: σ(x) = 1 / (1 + e^(-x))
    
    Rango: (0, 1)
    Uso: Capas ocultas, problemas de clasificación binaria
    
    Args:
        x: Valor o lista de valores de entrada
        
    Returns:
        Valor o lista de valores activados
    """
    if isinstance(x, list):
        # Clip para prevenir overflow
        x_clipped = clip_values(x)
        return [1.0 / (1.0 + math.exp(-val)) for val in x_clipped]
    else:
        x_clipped = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x_clipped))


def sigmoid_derivative(activated_output):
    """
    Derivada de la función Sigmoid.
    
    Formula: σ'(x) = σ(x) × (1 - σ(x))
    
    IMPORTANTE: Esta función recibe la salida YA ACTIVADA por sigmoid,
    no el valor pre-activación. Esto optimiza el cálculo durante backpropagation.
    
    Args:
        activated_output: Salida de sigmoid(x)
        
    Returns:
        Derivada evaluada en ese punto
    """
    if isinstance(activated_output, list):
        return [val * (1.0 - val) for val in activated_output]
    else:
        return activated_output * (1.0 - activated_output)


# ============================================================================
# TANH (Tangente Hiperbólica)
# ============================================================================

def tanh(x):
    """
    Función de activación Tangente Hiperbólica.
    
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Rango: (-1, 1)
    Uso: Capas ocultas (centrado en cero, mejor que sigmoid a veces)
    
    Args:
        x: Valor o lista de valores de entrada
        
    Returns:
        Valor o lista de valores activados
    """
    if isinstance(x, list):
        x_clipped = clip_values(x)
        result = []
        for val in x_clipped:
            exp_pos = math.exp(val)
            exp_neg = math.exp(-val)
            result.append((exp_pos - exp_neg) / (exp_pos + exp_neg))
        return result
    else:
        x_clipped = max(-500, min(500, x))
        exp_pos = math.exp(x_clipped)
        exp_neg = math.exp(-x_clipped)
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def tanh_derivative(activated_output):
    """
    Derivada de la función Tanh.
    
    Formula: tanh'(x) = 1 - tanh²(x)
    
    IMPORTANTE: Recibe la salida YA ACTIVADA por tanh.
    
    Args:
        activated_output: Salida de tanh(x)
        
    Returns:
        Derivada evaluada en ese punto
    """
    if isinstance(activated_output, list):
        return [1.0 - val**2 for val in activated_output]
    else:
        return 1.0 - activated_output**2


# ============================================================================
# RELU (Rectified Linear Unit)
# ============================================================================

def relu(x):
    """
    Función de activación ReLU.
    
    Formula: ReLU(x) = max(0, x)
    
    Rango: [0, ∞)
    Uso: Capas ocultas (muy popular en deep learning)
    Ventaja: No sufre de vanishing gradient
    
    Args:
        x: Valor o lista de valores de entrada
        
    Returns:
        Valor o lista de valores activados
    """
    if isinstance(x, list):
        return [max(0.0, val) for val in x]
    else:
        return max(0.0, x)


def relu_derivative(pre_activation):
    """
    Derivada de la función ReLU.
    
    Formula: ReLU'(x) = 1 if x > 0, else 0
    
    IMPORTANTE: A diferencia de sigmoid/tanh, esta función recibe
    el valor PRE-ACTIVACIÓN (antes de aplicar ReLU).
    
    Args:
        pre_activation: Valor z antes de la activación
        
    Returns:
        Derivada evaluada en ese punto
    """
    if isinstance(pre_activation, list):
        return [1.0 if val > 0 else 0.0 for val in pre_activation]
    else:
        return 1.0 if pre_activation > 0 else 0.0


# ============================================================================
# LEAKY RELU (Bonus)
# ============================================================================

def leaky_relu(x, alpha=0.01):
    """
    Función de activación Leaky ReLU.
    
    Formula: LeakyReLU(x) = x if x > 0, else alpha × x
    
    Rango: (-∞, ∞)
    Uso: Alternativa a ReLU que no "muere" para valores negativos
    
    Args:
        x: Valor o lista de valores de entrada
        alpha: Pendiente para valores negativos (típicamente 0.01)
        
    Returns:
        Valor o lista de valores activados
    """
    if isinstance(x, list):
        return [val if val > 0 else alpha * val for val in x]
    else:
        return x if x > 0 else alpha * x


def leaky_relu_derivative(pre_activation, alpha=0.01):
    """
    Derivada de Leaky ReLU.
    
    Formula: LeakyReLU'(x) = 1 if x > 0, else alpha
    
    Args:
        pre_activation: Valor z antes de la activación
        alpha: Pendiente para valores negativos
        
    Returns:
        Derivada evaluada en ese punto
    """
    if isinstance(pre_activation, list):
        return [1.0 if val > 0 else alpha for val in pre_activation]
    else:
        return 1.0 if pre_activation > 0 else alpha


# ============================================================================
# SOFTMAX (Para capa de salida)
# ============================================================================

def softmax(x):
    """
    Función de activación Softmax.
    
    Formula: softmax(xi) = e^xi / Σ(e^xj)
    
    Convierte un vector de valores en una distribución de probabilidad.
    La suma de todas las salidas es 1.0.
    
    Rango: (0, 1) para cada elemento, suma = 1
    Uso: Capa de salida para clasificación multiclase
    
    IMPORTANTE: Requerido por el subject para la output layer.
    
    Args:
        x: Lista de valores (logits)
        
    Returns:
        Lista de probabilidades que suman 1.0
    """
    if not isinstance(x, list):
        x = [x]
    
    # Truco numérico: restar el máximo para estabilidad (previene overflow)
    # softmax(x - c) = softmax(x) para cualquier constante c
    max_x = max(x)
    exp_values = [math.exp(val - max_x) for val in x]
    sum_exp = sum(exp_values)
    
    # Evitar división por cero (nunca debería pasar, pero por seguridad)
    if sum_exp == 0:
        sum_exp = 1e-10
    
    return [val / sum_exp for val in exp_values]


def softmax_derivative(activated_output):
    """
    Derivada de Softmax.
    
    NOTA: En la práctica, cuando se usa Softmax + Cross-Entropy,
    la derivada se simplifica a (ŷ - y) durante backpropagation.
    
    Esta función implementa la derivada completa del Jacobiano,
    pero normalmente no se usa directamente.
    
    Formula: ∂softmax(xi)/∂xj = softmax(xi) × (δij - softmax(xj))
    donde δ es el delta de Kronecker
    
    Args:
        activated_output: Salida de softmax(x)
        
    Returns:
        Matriz Jacobiana (lista de listas)
    """
    n = len(activated_output)
    jacobian = []
    
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                # Diagonal: si × (1 - si)
                derivative = activated_output[i] * (1.0 - activated_output[i])
            else:
                # Fuera de diagonal: -si × sj
                derivative = -activated_output[i] * activated_output[j]
            row.append(derivative)
        jacobian.append(row)
    
    return jacobian


# ============================================================================
# LINEAR (Identidad - para casos especiales)
# ============================================================================

def linear(x):
    """
    Función de activación Linear (identidad).
    
    Formula: f(x) = x
    
    Uso: Casos especiales, regresión
    
    Args:
        x: Valor o lista de valores
        
    Returns:
        El mismo valor/lista sin cambios
    """
    return x


def linear_derivative(x):
    """
    Derivada de la función Linear.
    
    Formula: f'(x) = 1
    
    Args:
        x: Valor o lista de valores
        
    Returns:
        1.0 o lista de 1.0s
    """
    if isinstance(x, list):
        return [1.0] * len(x)
    else:
        return 1.0


# ============================================================================
# DISPATCHER - Facilita acceso a funciones por nombre
# ============================================================================

ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'softmax': softmax,
    'linear': linear
}

ACTIVATION_DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative,
    'leaky_relu': leaky_relu_derivative,
    'softmax': softmax_derivative,
    'linear': linear_derivative
}


def get_activation(name):
    """
    Obtiene una función de activación por nombre.
    
    Args:
        name: Nombre de la función ('sigmoid', 'tanh', 'relu', etc.)
        
    Returns:
        Función de activación
    """
    name = name.lower()
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Función de activación desconocida: {name}")
    return ACTIVATION_FUNCTIONS[name]


def get_activation_derivative(name):
    """
    Obtiene la derivada de una función de activación por nombre.
    
    Args:
        name: Nombre de la función
        
    Returns:
        Función derivada
    """
    name = name.lower()
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Derivada de activación desconocida: {name}")
    return ACTIVATION_DERIVATIVES[name]


# ============================================================================
# TESTS Y VALIDACIÓN
# ============================================================================

def test_activations():
    """
    Tests básicos para validar las funciones de activación.
    Útil durante desarrollo.
    """
    print("🧪 Testeando funciones de activación...")
    print()
    
    # Test Sigmoid
    print("1️⃣  Sigmoid:")
    print(f"   sigmoid(0) = {sigmoid(0):.6f} (esperado: 0.5)")
    print(f"   sigmoid(1) = {sigmoid(1):.6f} (esperado: ~0.731)")
    print(f"   sigmoid(-1) = {sigmoid(-1):.6f} (esperado: ~0.269)")
    sig_out = sigmoid(0)
    print(f"   sigmoid'(0.5) = {sigmoid_derivative(sig_out):.6f} (esperado: 0.25)")
    print()
    
    # Test Tanh
    print("2️⃣  Tanh:")
    print(f"   tanh(0) = {tanh(0):.6f} (esperado: 0.0)")
    print(f"   tanh(1) = {tanh(1):.6f} (esperado: ~0.762)")
    print(f"   tanh(-1) = {tanh(-1):.6f} (esperado: ~-0.762)")
    tanh_out = tanh(0)
    print(f"   tanh'(0) = {tanh_derivative(tanh_out):.6f} (esperado: 1.0)")
    print()
    
    # Test ReLU
    print("3️⃣  ReLU:")
    print(f"   relu(5) = {relu(5):.6f} (esperado: 5.0)")
    print(f"   relu(0) = {relu(0):.6f} (esperado: 0.0)")
    print(f"   relu(-3) = {relu(-3):.6f} (esperado: 0.0)")
    print(f"   relu'(5) = {relu_derivative(5):.6f} (esperado: 1.0)")
    print(f"   relu'(-3) = {relu_derivative(-3):.6f} (esperado: 0.0)")
    print()
    
    # Test Softmax
    print("4️⃣  Softmax:")
    logits = [1.0, 2.0, 3.0]
    probs = softmax(logits)
    print(f"   softmax([1, 2, 3]) = [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"   Suma = {sum(probs):.6f} (esperado: 1.0)")
    print()
    
    # Test con listas
    print("5️⃣  Test con listas:")
    values = [0, 1, -1, 2, -2]
    sig_results = sigmoid(values)
    print(f"   sigmoid({values}) =")
    print(f"   [{', '.join(f'{v:.4f}' for v in sig_results)}]")
    print()
    
    print("✅ Todos los tests completados")
    print()


if __name__ == '__main__':
    # Ejecutar tests si se ejecuta directamente
    test_activations()
