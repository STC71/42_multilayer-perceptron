#!/usr/bin/env python3
"""
layers.py - Capas neuronales para el Multilayer Perceptron

Implementa la clase DenseLayer (fully connected layer) desde cero.

Una capa densa conecta todas las neuronas de la capa anterior con todas
las neuronas de la capa actual.

Operación:
    z = W · x + b        (weighted sum)
    a = activation(z)    (apply activation function)
    
Donde:
    - W: Matriz de pesos (n_out × n_in)
    - x: Vector de entrada (n_in)
    - b: Vector de bias (n_out)
    - z: Weighted sum (pre-activation)
    - a: Activación (output)
"""

from activation import get_activation, get_activation_derivative
from initialization import initialize_weights, initialize_bias


class DenseLayer:
    """
    Capa densa (fully connected) para red neuronal.
    
    Atributos:
        n_in: Número de neuronas en la capa anterior
        n_out: Número de neuronas en esta capa
        activation: Nombre de la función de activación
        weights: Matriz de pesos W (n_out × n_in)
        bias: Vector de bias b (n_out)
        
    Cache (para backpropagation):
        input: Entrada de la capa (x)
        z: Weighted sum (pre-activation)
        output: Salida activada (a)
    """
    
    def __init__(self, n_in, n_out, activation='sigmoid', 
                 weight_init='auto', bias_init='zeros', dropout_rate=0.0, seed=None):
        """
        Inicializa una capa densa.
        
        Args:
            n_in: Número de neuronas de entrada
            n_out: Número de neuronas de salida
            activation: Función de activación ('sigmoid', 'tanh', 'relu', 'softmax')
            weight_init: Método de inicialización de pesos
            bias_init: Método de inicialización de bias
            dropout_rate: Tasa de dropout (0.0-1.0), 0.0 = sin dropout (BONUS)
            seed: Semilla aleatoria (opcional)
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation.lower()
        self.dropout_rate = dropout_rate
        
        # Configurar generador aleatorio para dropout
        if seed is not None:
            import random
            self.random_state = random.Random(seed)
        else:
            import random
            self.random_state = random.Random()
        
        # Obtener funciones de activación
        self.activation_func = get_activation(self.activation_name)
        self.activation_deriv = get_activation_derivative(self.activation_name)
        
        # Inicializar pesos y bias
        self.weights = initialize_weights(
            n_in, n_out, 
            method=weight_init, 
            activation=self.activation_name,
            seed=seed
        )
        self.bias = initialize_bias(n_out, method=bias_init)
        
        # Cache para backpropagation
        self.input = None
        self.z = None  # Pre-activation (weighted sum)
        self.output = None  # Post-activation
        self.dropout_mask = None  # Máscara de dropout (BONUS)
        
        # Gradientes (calculados durante backpropagation)
        self.dW = None  # Gradiente respecto a W
        self.db = None  # Gradiente respecto a b
        self.dinput = None  # Gradiente respecto a la entrada
        
    def forward(self, x, training=False):
        """
        Propagación hacia adelante (feedforward).
        
        Calcula: a = activation(W · x + b)
        
        Si dropout_rate > 0 y training=True, aplica dropout después de activación.
        
        Args:
            x: Entrada de la capa (lista o lista de listas para batch)
                - Single sample: [x1, x2, ..., x_n_in]
                - Batch: [[x1, x2, ...], [x1, x2, ...], ...]
            training: Si es True, aplica dropout (BONUS)
        
        Returns:
            Salida activada (lista o lista de listas)
        """
        # Determinar si es batch o single sample
        is_batch = isinstance(x[0], list) if x else False
        
        if is_batch:
            return self._forward_batch(x, training)
        else:
            return self._forward_single(x, training)
    
    def _forward_single(self, x, training=False):
        """
        Forward pass para una sola muestra.
        
        Args:
            x: Vector de entrada (lista de tamaño n_in)
            training: Si True, aplica dropout
            
        Returns:
            Vector de salida activado (lista de tamaño n_out)
        """
        # Validar dimensiones
        if len(x) != self.n_in:
            raise ValueError(f"Entrada esperada de tamaño {self.n_in}, recibida {len(x)}")
        
        # Guardar entrada (necesaria para backprop)
        self.input = x
        
        # Calcular weighted sum: z = W · x + b
        z = []
        for i in range(self.n_out):
            # Producto punto entre fila i de W y el vector x
            weighted_sum = sum(self.weights[i][j] * x[j] for j in range(self.n_in))
            weighted_sum += self.bias[i]
            z.append(weighted_sum)
        
        # Guardar pre-activation (necesario para algunas derivadas)
        self.z = z
        
        # Aplicar activación
        if self.activation_name == 'softmax':
            # Softmax necesita todo el vector
            self.output = self.activation_func(z)
        else:
            # Otras activaciones son element-wise
            self.output = self.activation_func(z)
        
        # Aplicar dropout si está habilitado y en modo training (BONUS)
        if self.dropout_rate > 0.0 and training and self.activation_name != 'softmax':
            self.output, self.dropout_mask = self._apply_dropout(self.output)
        else:
            self.dropout_mask = None
        
        return self.output
    
    def _forward_batch(self, X, training=False):
        """
        Forward pass para un batch de muestras.
        
        Args:
            X: Batch de entradas (lista de listas, tamaño batch_size × n_in)
            training: Si True, aplica dropout
            
        Returns:
            Batch de salidas activadas (lista de listas, tamaño batch_size × n_out)
        """
        # Guardar entrada completa del batch
        self.input = X
        
        batch_size = len(X)
        Z = []  # Pre-activations para todo el batch
        outputs = []
        
        # Procesar cada muestra del batch
        for sample_idx in range(batch_size):
            x = X[sample_idx]
            
            # Validar dimensiones
            if len(x) != self.n_in:
                raise ValueError(f"Entrada esperada de tamaño {self.n_in}, recibida {len(x)}")
            
            # Calcular weighted sum para esta muestra
            z = []
            for i in range(self.n_out):
                weighted_sum = sum(self.weights[i][j] * x[j] for j in range(self.n_in))
                weighted_sum += self.bias[i]
                z.append(weighted_sum)
            
            Z.append(z)
        
        # Guardar pre-activations
        self.z = Z
        
        # Aplicar activación a cada muestra
        for z_sample in Z:
            if self.activation_name == 'softmax':
                output_sample = self.activation_func(z_sample)
            else:
                output_sample = self.activation_func(z_sample)
            outputs.append(output_sample)
        
        # Guardar salidas
        self.output = outputs
        
        # Aplicar dropout si está habilitado y en modo training (BONUS)
        if self.dropout_rate > 0.0 and training and self.activation_name != 'softmax':
            outputs_dropout = []
            dropout_masks = []
            for output_sample in outputs:
                output_d, mask_d = self._apply_dropout(output_sample)
                outputs_dropout.append(output_d)
                dropout_masks.append(mask_d)
            self.output = outputs_dropout
            self.dropout_mask = dropout_masks
        else:
            self.dropout_mask = None
        
        return self.output
    
    def get_output_dim(self):
        """Retorna las dimensiones de salida de esta capa."""
        return self.n_out
    
    def get_input_dim(self):
        """Retorna las dimensiones de entrada de esta capa."""
        return self.n_in
    
    def _apply_dropout(self, output):
        """
        Aplica dropout a la salida de la capa (BONUS).
        
        Dropout desactiva aleatoriamente neuronas durante training para prevenir
        overfitting. Durante inference, todas las neuronas están activas.
        
        Técnica:
            1. Generar máscara binaria: 1 con probabilidad (1-dropout_rate), 0 sino
            2. Multiplicar output por máscara (desactiva algunas neuronas)
            3. Escalar por 1/(1-dropout_rate) para mantener expectativa
        
        Inverted Dropout: Escalamos durante training, no durante inference
        
        Args:
            output: Salida activada de la capa
            
        Returns:
            Tupla (output_dropout, dropout_mask)
        """
        # Generar máscara binaria aleatoria
        # 1 con prob (1-dropout_rate), 0 con prob dropout_rate
        mask = [1.0 if self.random_state.random() > self.dropout_rate else 0.0 
                for _ in range(len(output))]
        
        # Aplicar máscara y escalar (inverted dropout)
        # Escalamos durante training para no tener que escalar durante inference
        scale = 1.0 / (1.0 - self.dropout_rate)
        output_dropout = [o * m * scale for o, m in zip(output, mask)]
        
        return output_dropout, mask
    
    def get_parameters_count(self):
        """
        Retorna el número total de parámetros entrenables en esta capa.
        
        Returns:
            Número de parámetros (weights + bias)
        """
        return (self.n_in * self.n_out) + self.n_out
    
    def __repr__(self):
        """Representación en string de la capa."""
        return (f"DenseLayer(in={self.n_in}, out={self.n_out}, "
                f"activation='{self.activation_name}', "
                f"params={self.get_parameters_count()})")


# ============================================================================
# TESTS
# ============================================================================

def test_dense_layer():
    """
    Tests básicos para validar DenseLayer.
    """
    print("🧪 Testeando DenseLayer...")
    print()
    
    # Crear una capa pequeña
    print("1️⃣  Creando capa: 3 → 2 (sigmoid)")
    layer = DenseLayer(n_in=3, n_out=2, activation='sigmoid', seed=42)
    print(f"   {layer}")
    print(f"   Parámetros: {layer.get_parameters_count()}")
    print()
    
    # Test forward single sample
    print("2️⃣  Forward pass (single sample):")
    x = [1.0, 2.0, 3.0]
    output = layer.forward(x)
    print(f"   Input:  {x}")
    print(f"   Output: [{', '.join(f'{v:.4f}' for v in output)}]")
    print(f"   Sum:    {sum(output):.4f} (para sigmoid, suma ≠ 1)")
    print()
    
    # Test forward batch
    print("3️⃣  Forward pass (batch):")
    X = [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [-1.0, 0.0, 1.0]]
    outputs = layer.forward(X)
    print(f"   Batch size: {len(X)}")
    print(f"   Outputs:")
    for i, out in enumerate(outputs):
        print(f"      Sample {i}: [{', '.join(f'{v:.4f}' for v in out)}]")
    print()
    
    # Test con softmax
    print("4️⃣  Capa con Softmax:")
    layer_softmax = DenseLayer(n_in=3, n_out=2, activation='softmax', seed=42)
    output_sm = layer_softmax.forward(x)
    print(f"   Output: [{', '.join(f'{v:.4f}' for v in output_sm)}]")
    print(f"   Sum:    {sum(output_sm):.6f} (para softmax, suma = 1.0)")
    print()
    
    print("✅ Todos los tests completados")
    print()


if __name__ == '__main__':
    test_dense_layer()
