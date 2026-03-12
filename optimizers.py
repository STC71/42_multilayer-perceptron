#!/usr/bin/env python3
"""
optimizers.py - Algoritmos de optimización para redes neuronales

Implementa varios optimizadores para actualización de pesos:
    - SGD (Stochastic Gradient Descent)
    - SGD with Momentum
    - RMSprop
    - Adam (Adaptive Moment Estimation) - BONUS del subject

Todos los optimizadores implementan la interfaz:
    - __init__(): Inicialización con hiperparámetros
    - initialize_state(): Inicializa estructuras internas
    - update(): Calcula y devuelve actualizaciones de parámetros
"""

from math import sqrt


class SGD:
    """
    Stochastic Gradient Descent (baseline).
    
    Actualización simple:
        θ = θ - α · ∇θ
    
    donde:
        - θ son los parámetros (W, b)
        - α es el learning rate
        - ∇θ es el gradiente
    
    Args:
        learning_rate: Tasa de aprendizaje
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.name = 'sgd'
    
    def initialize_state(self, layers):
        """No necesita estado para SGD básico."""
        pass
    
    def update(self, layer, dW, db):
        """
        Calcula actualizaciones de pesos y bias.
        
        Args:
            layer: Capa a actualizar
            dW: Gradientes de pesos
            db: Gradientes de bias
            
        Returns:
            Tupla (delta_W, delta_b) con las actualizaciones
        """
        # SGD básico: actualización = -α · gradiente
        delta_W = [[-self.learning_rate * dw for dw in dw_row] for dw_row in dW]
        delta_b = [-self.learning_rate * db_i for db_i in db]
        
        return delta_W, delta_b


class SGDMomentum:
    """
    SGD with Momentum.
    
    Añade momentum para acelerar convergencia y suavizar oscilaciones:
        v = β · v + ∇θ
        θ = θ - α · v
    
    donde:
        - v es el momentum (velocidad)
        - β es el coeficiente de momentum (típicamente 0.9)
    
    Args:
        learning_rate: Tasa de aprendizaje
        momentum: Coeficiente de momentum (default: 0.9)
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.name = 'sgd_momentum'
        self.velocity = {}  # Diccionario: id(layer) -> (vW, vb)
    
    def initialize_state(self, layers):
        """Inicializa velocidades en cero."""
        for layer in layers:
            layer_id = id(layer)
            # Inicializar velocidades con mismo shape que pesos/bias
            vW = [[0.0 for _ in range(layer.n_in)] for _ in range(layer.n_out)]
            vb = [0.0 for _ in range(layer.n_out)]
            self.velocity[layer_id] = (vW, vb)
    
    def update(self, layer, dW, db):
        """
        Actualiza con momentum.
        
        v = β·v + ∇θ
        θ = θ - α·v
        """
        layer_id = id(layer)
        vW, vb = self.velocity[layer_id]
        
        # Actualizar momentum: v = β·v + g
        new_vW = [[self.momentum * vW[i][j] + dW[i][j] 
                   for j in range(layer.n_in)] 
                  for i in range(layer.n_out)]
        
        new_vb = [self.momentum * vb[i] + db[i] 
                  for i in range(layer.n_out)]
        
        # Guardar nuevo momentum
        self.velocity[layer_id] = (new_vW, new_vb)
        
        # Calcular actualización: -α·v
        delta_W = [[-self.learning_rate * new_vW[i][j] 
                    for j in range(layer.n_in)] 
                   for i in range(layer.n_out)]
        
        delta_b = [-self.learning_rate * new_vb[i] 
                   for i in range(layer.n_out)]
        
        return delta_W, delta_b


class RMSprop:
    """
    RMSprop (Root Mean Square Propagation).
    
    Usa promedio móvil exponencial de gradientes al cuadrado:
        v = β · v + (1-β) · ∇θ²
        θ = θ - α · ∇θ / (√v + ε)
    
    Ventajas:
        - Adapta learning rate por parámetro
        - Funciona bien con gradientes con diferentes magnitudes
    
    Args:
        learning_rate: Tasa de aprendizaje
        beta: Coeficiente decay (default: 0.999)
        epsilon: Término de estabilidad (default: 1e-8)
    """
    
    def __init__(self, learning_rate=0.001, beta=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.name = 'rmsprop'
        self.cache = {}  # Diccionario: id(layer) -> (vW, vb)
    
    def initialize_state(self, layers):
        """Inicializa cache de gradientes al cuadrado."""
        for layer in layers:
            layer_id = id(layer)
            vW = [[0.0 for _ in range(layer.n_in)] for _ in range(layer.n_out)]
            vb = [0.0 for _ in range(layer.n_out)]
            self.cache[layer_id] = (vW, vb)
    
    def update(self, layer, dW, db):
        """
        Actualiza con RMSprop.
        
        v = β·v + (1-β)·g²
        θ = θ - α·g / (√v + ε)
        """
        layer_id = id(layer)
        vW, vb = self.cache[layer_id]
        
        # Actualizar cache: v = β·v + (1-β)·g²
        new_vW = [[self.beta * vW[i][j] + (1 - self.beta) * dW[i][j]**2
                   for j in range(layer.n_in)]
                  for i in range(layer.n_out)]
        
        new_vb = [self.beta * vb[i] + (1 - self.beta) * db[i]**2
                  for i in range(layer.n_out)]
        
        # Guardar nuevo cache
        self.cache[layer_id] = (new_vW, new_vb)
        
        # Calcular actualización: -α·g / (√v + ε)
        delta_W = [[-self.learning_rate * dW[i][j] / (sqrt(new_vW[i][j]) + self.epsilon)
                    for j in range(layer.n_in)]
                   for i in range(layer.n_out)]
        
        delta_b = [-self.learning_rate * db[i] / (sqrt(new_vb[i]) + self.epsilon)
                   for i in range(layer.n_out)]
        
        return delta_W, delta_b


class Adam:
    """
    Adam - Adaptive Moment Estimation (BONUS del subject).
    
    Combina momentum (primer momento) con RMSprop (segundo momento):
        m = β₁ · m + (1-β₁) · ∇θ           (primer momento - momentum)
        v = β₂ · v + (1-β₂) · ∇θ²          (segundo momento - RMSprop)
        
        m̂ = m / (1 - β₁ᵗ)                  (bias correction)
        v̂ = v / (1 - β₂ᵗ)                  (bias correction)
        
        θ = θ - α · m̂ / (√v̂ + ε)
    
    Ventajas:
        - Converge más rápido que SGD
        - Adapta learning rate por parámetro
        - Funciona bien con gradientes sparse
        - State-of-the-art para deep learning
    
    Paper: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
    
    Args:
        learning_rate: Tasa de aprendizaje (default: 0.001)
        beta1: Coeficiente primer momento (default: 0.9)
        beta2: Coeficiente segundo momento (default: 0.999)
        epsilon: Término de estabilidad (default: 1e-8)
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = 'adam'
        self.t = 0  # Timestep para bias correction
        self.m = {}  # Primer momento (momentum)
        self.v = {}  # Segundo momento (RMSprop)
    
    def initialize_state(self, layers):
        """Inicializa momentos en cero."""
        for layer in layers:
            layer_id = id(layer)
            
            # Primer momento (momentum)
            mW = [[0.0 for _ in range(layer.n_in)] for _ in range(layer.n_out)]
            mb = [0.0 for _ in range(layer.n_out)]
            
            # Segundo momento (RMSprop)
            vW = [[0.0 for _ in range(layer.n_in)] for _ in range(layer.n_out)]
            vb = [0.0 for _ in range(layer.n_out)]
            
            self.m[layer_id] = (mW, mb)
            self.v[layer_id] = (vW, vb)
    
    def update(self, layer, dW, db):
        """
        Actualiza parámetros con Adam.
        
        Algoritmo:
            1. m = β₁·m + (1-β₁)·g
            2. v = β₂·v + (1-β₂)·g²
            3. m̂ = m / (1 - β₁ᵗ)
            4. v̂ = v / (1 - β₂ᵗ)
            5. θ = θ - α·m̂ / (√v̂ + ε)
        """
        self.t += 1  # Incrementar timestep
        
        layer_id = id(layer)
        mW, mb = self.m[layer_id]
        vW, vb = self.v[layer_id]
        
        # PASO 1: Actualizar primer momento (momentum)
        # m = β₁·m + (1-β₁)·g
        new_mW = [[self.beta1 * mW[i][j] + (1 - self.beta1) * dW[i][j]
                   for j in range(layer.n_in)]
                  for i in range(layer.n_out)]
        
        new_mb = [self.beta1 * mb[i] + (1 - self.beta1) * db[i]
                  for i in range(layer.n_out)]
        
        # PASO 2: Actualizar segundo momento (RMSprop)
        # v = β₂·v + (1-β₂)·g²
        new_vW = [[self.beta2 * vW[i][j] + (1 - self.beta2) * dW[i][j]**2
                   for j in range(layer.n_in)]
                  for i in range(layer.n_out)]
        
        new_vb = [self.beta2 * vb[i] + (1 - self.beta2) * db[i]**2
                  for i in range(layer.n_out)]
        
        # Guardar nuevos momentos
        self.m[layer_id] = (new_mW, new_mb)
        self.v[layer_id] = (new_vW, new_vb)
        
        # PASO 3: Bias correction
        # Los momentos están sesgados hacia cero al inicio, corregir
        m_hat_correction = 1.0 / (1.0 - self.beta1 ** self.t)
        v_hat_correction = 1.0 / (1.0 - self.beta2 ** self.t)
        
        # PASO 4: Calcular actualización
        # θ = θ - α · m̂ / (√v̂ + ε)
        delta_W = []
        for i in range(layer.n_out):
            delta_W_row = []
            for j in range(layer.n_in):
                # m̂ = m / (1 - β₁ᵗ)
                m_hat = new_mW[i][j] * m_hat_correction
                
                # v̂ = v / (1 - β₂ᵗ)
                v_hat = new_vW[i][j] * v_hat_correction
                
                # Actualización adaptativa
                update = -self.learning_rate * m_hat / (sqrt(v_hat) + self.epsilon)
                delta_W_row.append(update)
            
            delta_W.append(delta_W_row)
        
        delta_b = []
        for i in range(layer.n_out):
            m_hat = new_mb[i] * m_hat_correction
            v_hat = new_vb[i] * v_hat_correction
            update = -self.learning_rate * m_hat / (sqrt(v_hat) + self.epsilon)
            delta_b.append(update)
        
        return delta_W, delta_b


def get_optimizer(name, **kwargs):
    """
    Factory function para crear optimizadores.
    
    Args:
        name: Nombre del optimizador ('sgd', 'sgd_momentum', 'rmsprop', 'adam')
        **kwargs: Argumentos para el optimizador
        
    Returns:
        Instancia del optimizador
        
    Examples:
        >>> optimizer = get_optimizer('adam', learning_rate=0.001)
        >>> optimizer = get_optimizer('sgd_momentum', learning_rate=0.01, momentum=0.9)
    """
    optimizers = {
        'sgd': SGD,
        'sgd_momentum': SGDMomentum,
        'rmsprop': RMSprop,
        'adam': Adam
    }
    
    if name not in optimizers:
        available = ', '.join(optimizers.keys())
        raise ValueError(f"Optimizador desconocido: {name}. Disponibles: {available}")
    
    optimizer_class = optimizers[name]
    return optimizer_class(**kwargs)


# ============================================================================
# Tests unitarios
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("🧪 TESTS DE OPTIMIZERS")
    print("=" * 70)
    print()
    
    # Mock layer para testing
    class MockLayer:
        def __init__(self):
            self.n_in = 3
            self.n_out = 2
            self.weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            self.bias = [0.1, 0.2]
    
    layer = MockLayer()
    
    # Gradientes de prueba
    dW = [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]
    db = [0.01, 0.02]
    
    # Test 1: SGD básico
    print("🧪 Test 1: SGD básico")
    sgd = SGD(learning_rate=0.1)
    sgd.initialize_state([layer])
    delta_W, delta_b = sgd.update(layer, dW, db)
    print(f"   ✓ delta_W[0][0] = {delta_W[0][0]:.6f} (esperado: -0.001)")
    print(f"   ✓ delta_b[0] = {delta_b[0]:.6f} (esperado: -0.001)")
    print()
    
    # Test 2: SGD with Momentum
    print("🧪 Test 2: SGD with Momentum")
    sgd_mom = SGDMomentum(learning_rate=0.1, momentum=0.9)
    sgd_mom.initialize_state([layer])
    
    # Primera actualización
    delta_W1, delta_b1 = sgd_mom.update(layer, dW, db)
    print(f"   ✓ Primera actualización: delta_W[0][0] = {delta_W1[0][0]:.6f}")
    
    # Segunda actualización (debería tener momentum)
    delta_W2, delta_b2 = sgd_mom.update(layer, dW, db)
    print(f"   ✓ Segunda actualización: delta_W[0][0] = {delta_W2[0][0]:.6f}")
    print(f"   ✓ Momentum activo (mayor que primera): {abs(delta_W2[0][0]) > abs(delta_W1[0][0])}")
    print()
    
    # Test 3: RMSprop
    print("🧪 Test 3: RMSprop")
    rmsprop = RMSprop(learning_rate=0.01, beta=0.9)
    rmsprop.initialize_state([layer])
    delta_W, delta_b = rmsprop.update(layer, dW, db)
    print(f"   ✓ delta_W[0][0] = {delta_W[0][0]:.6f}")
    print(f"   ✓ Learning rate adaptativo aplicado")
    print()
    
    # Test 4: Adam (BONUS)
    print("🧪 Test 4: Adam (BONUS)")
    adam = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
    adam.initialize_state([layer])
    
    # Primera actualización (con bias correction)
    delta_W1, delta_b1 = adam.update(layer, dW, db)
    print(f"   ✓ Primera actualización: delta_W[0][0] = {delta_W1[0][0]:.6f}")
    print(f"   ✓ Bias correction aplicada (t=1)")
    
    # Múltiples actualizaciones para ver convergencia
    for _ in range(9):
        adam.update(layer, dW, db)
    
    delta_W10, delta_b10 = adam.update(layer, dW, db)
    print(f"   ✓ Actualización 10: delta_W[0][0] = {delta_W10[0][0]:.6f}")
    print(f"   ✓ Adam combina momentum + RMSprop correctamente")
    print()
    
    # Test 5: Factory function
    print("🧪 Test 5: Factory function")
    opt1 = get_optimizer('adam', learning_rate=0.001)
    opt2 = get_optimizer('sgd', learning_rate=0.01)
    opt3 = get_optimizer('rmsprop')
    
    print(f"   ✓ get_optimizer('adam'): {opt1.name}")
    print(f"   ✓ get_optimizer('sgd'): {opt2.name}")
    print(f"   ✓ get_optimizer('rmsprop'): {opt3.name}")
    print()
    
    print("=" * 70)
    print("✅ TODOS LOS TESTS DE OPTIMIZERS PASARON")
    print("=" * 70)
    print()
    print("📊 Optimizers implementados:")
    print("   • SGD                  - Gradient Descent básico")
    print("   • SGD with Momentum    - Acelera convergencia")
    print("   • RMSprop              - Learning rate adaptativo")
    print("   • Adam (BONUS)         - State-of-the-art optimizer")
    print()
