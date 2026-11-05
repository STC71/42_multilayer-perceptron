#!/usr/bin/env python3
"""
network.py - Clase principal del Multilayer Perceptron

Implementa la red neuronal completa con:
    - Feedforward: Propagación hacia adelante
    - Backpropagation: Cálculo de gradientes
    - Gradient Descent: Actualización de pesos
    
ALGORITMOS CLAVE (según el subject):

1. FEEDFORWARD:
   Para cada capa l:
       z[l] = W[l] · a[l-1] + b[l]
       a[l] = activation(z[l])

2. BACKPROPAGATION:
   a) Error en output layer:
      δ[L] = ∂Loss/∂a[L] ⊙ activation'(z[L])
   
   b) Propagar error hacia atrás:
      Para l = L-1 hasta 1:
          δ[l] = (W[l+1]ᵀ · δ[l+1]) ⊙ activation'(z[l])
   
   c) Calcular gradientes:
      ∂Loss/∂W[l] = δ[l] · a[l-1]ᵀ
      ∂Loss/∂b[l] = δ[l]

3. GRADIENT DESCENT:
   W[l] = W[l] - α · ∂Loss/∂W[l]
   b[l] = b[l] - α · ∂Loss/∂b[l]
   
   donde α es el learning rate
"""

from layers import DenseLayer
from loss import get_loss_function, get_loss_derivative
from optimizers import get_optimizer
import sys


class Network:
    """
    Red neuronal Multilayer Perceptron.
    
    Gestiona un stack de capas densas y ejecuta el ciclo completo de
    entrenamiento: feedforward, backpropagation y actualización de pesos.
    
    Soporta múltiples optimizadores (BONUS):
        - SGD (Gradient Descent básico)
        - SGD with Momentum
        - RMSprop
        - Adam (state-of-the-art)
    
    Atributos:
        layers: Lista de DenseLayer
        loss_function: Función de pérdida
        learning_rate: Tasa de aprendizaje
        optimizer: Optimizador para actualización de pesos (BONUS)
    """
    
    def __init__(self, loss='binary_crossentropy', learning_rate=0.01, optimizer='sgd'):
        """
        Inicializa la red neuronal.
        
        Args:
            loss: Nombre de la función de pérdida
            learning_rate: Tasa de aprendizaje
            optimizer: Optimizador ('sgd', 'sgd_momentum', 'rmsprop', 'adam')
        """
        self.layers = []
        self.loss_name = loss.lower()
        self.loss_function = get_loss_function(self.loss_name)
        self.loss_derivative = get_loss_derivative(self.loss_name)
        self.learning_rate = learning_rate
        
        # Inicializar optimizer (BONUS)
        if optimizer == 'sgd':
            self.optimizer = get_optimizer('sgd', learning_rate=learning_rate)
        elif optimizer == 'adam':
            # Adam usa learning rate más bajo por defecto
            adam_lr = learning_rate if learning_rate < 0.01 else 0.001
            self.optimizer = get_optimizer('adam', learning_rate=adam_lr)
        elif optimizer == 'rmsprop':
            rmsprop_lr = learning_rate if learning_rate < 0.01 else 0.001
            self.optimizer = get_optimizer('rmsprop', learning_rate=rmsprop_lr)
        elif optimizer == 'sgd_momentum':
            self.optimizer = get_optimizer('sgd_momentum', learning_rate=learning_rate, momentum=0.9)
        else:
            print(f"⚠️  Optimizer '{optimizer}' desconocido, usando SGD", file=sys.stderr)
            self.optimizer = get_optimizer('sgd', learning_rate=learning_rate)
        
        # Historia de entrenamiento
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
    
    def add_layer(self, layer):
        """
        Añade una capa a la red.
        
        Args:
            layer: Instancia de DenseLayer
        """
        # Validar dimensiones con la capa anterior
        if self.layers:
            prev_layer = self.layers[-1]
            if layer.n_in != prev_layer.n_out:
                raise ValueError(
                    f"Dimensiones incompatibles: capa anterior salida={prev_layer.n_out}, "
                    f"nueva capa entrada={layer.n_in}"
                )
        
        self.layers.append(layer)
    
    def build(self, architecture, activations=None, weight_init='he', dropout_rate=0.0, seed=None):
        """
        Construye la red con una arquitectura especificada.
        
        Args:
            architecture: Lista con número de neuronas por capa [30, 24, 24, 2]
            activations: Lista de activaciones por capa (None = sigmoid por defecto)
            weight_init: Método de inicialización de pesos
            dropout_rate: Tasa de dropout para capas ocultas (0.0-1.0) - BONUS
            seed: Semilla aleatoria
            
        Ejemplo:
            network.build([30, 24, 24, 2], ['relu', 'relu', 'softmax'], dropout_rate=0.2)
        """
        if len(architecture) < 2:
            raise ValueError("La arquitectura debe tener al menos 2 capas (input y output)")
        
        # Activaciones por defecto
        if activations is None:
            activations = ['sigmoid'] * (len(architecture) - 2) + ['softmax']
        
        if len(activations) != len(architecture) - 1:
            raise ValueError(
                f"Número de activaciones ({len(activations)}) debe ser "
                f"igual a número de capas - 1 ({len(architecture) - 1})"
            )
        
        # Crear capas
        self.layers = []
        for i in range(len(architecture) - 1):
            n_in = architecture[i]
            n_out = architecture[i + 1]
            activation = activations[i]
            
            # Dropout solo en capas ocultas, no en output (BONUS)
            is_output_layer = (i == len(architecture) - 2)
            layer_dropout = 0.0 if is_output_layer else dropout_rate
            
            layer = DenseLayer(
                n_in=n_in,
                n_out=n_out,
                activation=activation,
                weight_init=weight_init,
                dropout_rate=layer_dropout,
                seed=seed
            )
            
            self.layers.append(layer)
        
        # Inicializar estado del optimizer (BONUS)
        self.optimizer.initialize_state(self.layers)
    
    def forward(self, x, training=False):
        """
        Propagación hacia adelante a través de todas las capas.
        
        Args:
            x: Entrada (single sample o batch)
                - Single: [x1, x2, ..., x_n]
                - Batch: [[x1, x2, ...], [x1, x2, ...], ...]
            training: Si True, aplica dropout en capas ocultas (BONUS)
        
        Returns:
            Salida de la red (última capa activada)
        """
        activation = x
        
        # Propagar a través de cada capa
        for layer in self.layers:
            activation = layer.forward(activation, training=training)
        
        return activation
    
    def predict(self, x):
        """
        Realiza predicción sin guardar cache (modo inferencia).
        
        Args:
            x: Entrada (single o batch)
            
        Returns:
            Predicciones
        """
        return self.forward(x)
    
    def backward(self, y_true, y_pred):
        """
        Backpropagation: Calcula gradientes para todas las capas.
        
        ALGORITMO (del subject):
        1. Calcular error en output layer
        2. Propagar error hacia atrás capa por capa
        3. Calcular gradientes ∂Loss/∂W y ∂Loss/∂b para cada capa
        
        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones de la red
        """
        # Determinar si es batch o single sample
        is_batch = isinstance(y_pred[0], list) if y_pred and len(y_pred) > 0 else False
        
        if is_batch:
            self._backward_batch(y_true, y_pred)
        else:
            self._backward_single(y_true, y_pred)
    
    def _backward_single(self, y_true, y_pred):
        """
        Backpropagation para una sola muestra.
        
        Args:
            y_true: Label verdadero
            y_pred: Predicción
        """
        n_layers = len(self.layers)
        
        # PASO 1: Calcular error en la última capa (output layer)
        # Para softmax + categorical cross-entropy: δ = ŷ - y (simplificación)
        # Para sigmoid + binary cross-entropy: δ = (ŷ - y) dependiendo de derivadas
        
        output_layer = self.layers[-1]
        
        # Simplificación para softmax + cross-entropy
        if output_layer.activation_name == 'softmax':
            # Convertir y_true a one-hot si es un escalar
            if not isinstance(y_true, list):
                y_true_onehot = [0.0] * output_layer.n_out
                y_true_onehot[int(y_true)] = 1.0
                y_true = y_true_onehot
            
            # δ[L] = ŷ - y (derivada simplificada de softmax + cross-entropy)
            delta = [yp - yt for yp, yt in zip(y_pred, y_true)]
        else:
            # Para otras activaciones: usar derivada de loss × derivada de activación
            loss_grad = self.loss_derivative([y_true] if not isinstance(y_true, list) else y_true, 
                                            [y_pred] if not isinstance(y_pred, list) else y_pred)
            
            # Derivada de la activación
            if output_layer.activation_name in ['relu', 'leaky_relu']:
                activation_grad = output_layer.activation_deriv(output_layer.z)
            else:
                activation_grad = output_layer.activation_deriv(output_layer.output)
            
            # δ = ∂Loss/∂a ⊙ ∂a/∂z
            delta = [lg * ag for lg, ag in zip(loss_grad, activation_grad)]
        
        output_layer.delta = delta
        
        # PASO 2: Propagar error hacia atrás (backward pass)
        for l in range(n_layers - 2, -1, -1):
            current_layer = self.layers[l]
            next_layer = self.layers[l + 1]
            
            # δ[l] = (W[l+1]ᵀ · δ[l+1]) ⊙ activation'(z[l])
            delta_next = next_layer.delta
            W_next = next_layer.weights
            
            # W[l+1]ᵀ · δ[l+1]
            delta_propagated = []
            for j in range(current_layer.n_out):
                sum_val = sum(W_next[k][j] * delta_next[k] for k in range(next_layer.n_out))
                delta_propagated.append(sum_val)
            
            # ⊙ activation'(z[l])
            if current_layer.activation_name in ['relu', 'leaky_relu']:
                activation_grad = current_layer.activation_deriv(current_layer.z)
            else:
                activation_grad = current_layer.activation_deriv(current_layer.output)

            delta = [dp * ag for dp, ag in zip(delta_propagated, activation_grad)]
            if current_layer.dropout_mask is not None:
                scale = 1.0 / (1.0 - current_layer.dropout_rate)
                delta = [
                    value * mask * scale
                    for value, mask in zip(delta, current_layer.dropout_mask)
                ]
            current_layer.delta = delta
        
        # PASO 3: Calcular gradientes para pesos y bias
        for l, layer in enumerate(self.layers):
            # Determinar la entrada de esta capa
            if l == 0:
                layer_input = layer.input
            else:
                layer_input = self.layers[l - 1].output
            
            # ∂Loss/∂W[l] = δ[l] · a[l-1]ᵀ
            dW = []
            for i in range(layer.n_out):
                dW_row = []
                for j in range(layer.n_in):
                    gradient = layer.delta[i] * layer_input[j]
                    dW_row.append(gradient)
                dW.append(dW_row)
            
            # ∂Loss/∂b[l] = δ[l]
            db = layer.delta.copy()
            
            # Guardar gradientes
            layer.dW = dW
            layer.db = db
    
    def _backward_batch(self, y_true, y_pred):
        """
        Backpropagation para un batch de muestras.
        
        Los gradientes se promedian sobre el batch.
        
        Args:
            y_true: Batch de labels verdaderos
            y_pred: Batch de predicciones
        """
        batch_size = len(y_true)
        n_layers = len(self.layers)
        
        # Inicializar acumuladores de deltas para todo el batch
        batch_deltas = [[] for _ in range(n_layers)]
        
        # Procesar cada muestra del batch
        for sample_idx in range(batch_size):
            yt_sample = y_true[sample_idx]
            yp_sample = y_pred[sample_idx]
            
            # PASO 1: Error en output layer para esta muestra
            output_layer = self.layers[-1]
            
            if output_layer.activation_name == 'softmax':
                # Convertir y_true a one-hot si es necesario
                if not isinstance(yt_sample, list):
                    yt_onehot = [0.0] * output_layer.n_out
                    yt_onehot[int(yt_sample)] = 1.0
                    yt_sample = yt_onehot
                
                # δ = ŷ - y
                delta = [yp - yt for yp, yt in zip(yp_sample, yt_sample)]
            else:
                # Usar derivadas explícitas
                z_sample = output_layer.z[sample_idx] if isinstance(output_layer.z[0], list) else output_layer.z
                output_sample = output_layer.output[sample_idx] if isinstance(output_layer.output[0], list) else output_layer.output
                
                loss_grad = self.loss_derivative([yt_sample], [yp_sample])
                
                if output_layer.activation_name in ['relu', 'leaky_relu']:
                    activation_grad = output_layer.activation_deriv(z_sample)
                else:
                    activation_grad = output_layer.activation_deriv(output_sample)
                
                delta = [lg * ag for lg, ag in zip(loss_grad, activation_grad)]
            
            batch_deltas[-1].append(delta)
            
            # PASO 2: Propagar hacia atrás
            for l in range(n_layers - 2, -1, -1):
                current_layer = self.layers[l]
                next_layer = self.layers[l + 1]
                
                # Obtener delta de la siguiente capa para esta muestra
                delta_next = batch_deltas[l + 1][sample_idx]
                W_next = next_layer.weights
                
                # W[l+1]ᵀ · δ[l+1]
                delta_propagated = []
                for j in range(current_layer.n_out):
                    sum_val = sum(W_next[k][j] * delta_next[k] for k in range(next_layer.n_out))
                    delta_propagated.append(sum_val)
                
                # Obtener valores necesarios para esta muestra
                z_sample = current_layer.z[sample_idx] if isinstance(current_layer.z[0], list) else current_layer.z
                output_sample = current_layer.output[sample_idx] if isinstance(current_layer.output[0], list) else current_layer.output
                
                # ⊙ activation'(z[l])
                if current_layer.activation_name in ['relu', 'leaky_relu']:
                    activation_grad = current_layer.activation_deriv(z_sample)
                else:
                    activation_grad = current_layer.activation_deriv(output_sample)

                delta = [dp * ag for dp, ag in zip(delta_propagated, activation_grad)]
                if current_layer.dropout_mask is not None:
                    scale = 1.0 / (1.0 - current_layer.dropout_rate)
                    dropout_mask = current_layer.dropout_mask[sample_idx]
                    delta = [
                        value * mask * scale
                        for value, mask in zip(delta, dropout_mask)
                    ]
                batch_deltas[l].append(delta)
        
        # PASO 3: Promediar gradientes sobre el batch
        for l, layer in enumerate(self.layers):
            # Obtener inputs de esta capa para todo el batch
            if l == 0:
                layer_inputs = layer.input
            else:
                layer_inputs = self.layers[l - 1].output
            
            # Inicializar acumuladores de gradientes
            dW = [[0.0] * layer.n_in for _ in range(layer.n_out)]
            db = [0.0] * layer.n_out
            
            # Acumular gradientes de todas las muestras del batch
            for sample_idx in range(batch_size):
                delta_sample = batch_deltas[l][sample_idx]
                input_sample = layer_inputs[sample_idx] if isinstance(layer_inputs[0], list) else layer_inputs
                
                # ∂Loss/∂W[l] = δ[l] · a[l-1]ᵀ
                for i in range(layer.n_out):
                    for j in range(layer.n_in):
                        dW[i][j] += delta_sample[i] * input_sample[j]
                
                # ∂Loss/∂b[l] = δ[l]
                for i in range(layer.n_out):
                    db[i] += delta_sample[i]
            
            # Promediar sobre el batch
            dW = [[grad / batch_size for grad in row] for row in dW]
            db = [grad / batch_size for grad in db]
            
            # Guardar gradientes
            layer.dW = dW
            layer.db = db
    
    def update_weights(self):
        """
        Actualiza pesos usando el optimizer configurado (BONUS: Adam/RMSprop/Momentum).
        
        El optimizer calcula las actualizaciones basándose en:
            - SGD: θ = θ - α · ∇θ
            - Momentum: v = β·v + ∇θ, θ = θ - α·v
            - RMSprop: v = β·v + (1-β)·∇θ², θ = θ - α·∇θ/(√v + ε)
            - Adam: combina momentum + RMSprop con bias correction
        
        donde:
            - θ representa W o b
            - ∇θ son los gradientes (dW, db)
        """
        for layer in self.layers:
            # Calcular actualizaciones usando optimizer (BONUS)
            delta_W, delta_b = self.optimizer.update(layer, layer.dW, layer.db)
            
            # Aplicar actualizaciones: W = W + delta_W (delta ya incluye el signo)
            for i in range(layer.n_out):
                for j in range(layer.n_in):
                    layer.weights[i][j] += delta_W[i][j]
            
            # Actualizar bias: b = b + delta_b
            for i in range(layer.n_out):
                layer.bias[i] += delta_b[i]
    
    def train_step(self, X, y):
        """
        Ejecuta un paso completo de entrenamiento.
        
        1. Forward pass (con dropout si está habilitado)
        2. Calcular pérdida
        3. Backward pass (backpropagation)
        4. Actualizar pesos (usando optimizer)
        
        Args:
            X: Batch de inputs
            y: Batch de labels
            
        Returns:
            Valor de pérdida
        """
        # 1. Forward pass con training=True (activa dropout) - BONUS
        predictions = self.forward(X, training=True)
        
        # 2. Calcular pérdida
        loss_value = self.compute_loss(y, predictions)
        
        # 3. Backpropagation
        self.backward(y, predictions)
        
        # 4. Actualizar pesos con optimizer (BONUS: Adam/RMSprop/Momentum)
        self.update_weights()
        
        return loss_value
    
    def compute_loss(self, y_true, y_pred):
        """
        Calcula la pérdida entre predicciones y valores verdaderos.
        
        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones
            
        Returns:
            Valor de pérdida
        """
        # Para output con softmax (2 clases), extraer probabilidad de clase positiva
        if isinstance(y_pred[0], list):
            # Batch de predicciones con one-hot
            if self.loss_name == 'binary_crossentropy':
                # Convertir de formato one-hot a probabilidad única
                y_pred_probs = [pred[1] for pred in y_pred]  # Probabilidad de clase 1 (Maligno)
                return self.loss_function(y_true, y_pred_probs)
            else:
                # Categorical cross-entropy necesita one-hot
                y_true_onehot = []
                for yt in y_true:
                    onehot = [0.0] * len(y_pred[0])
                    onehot[int(yt)] = 1.0
                    y_true_onehot.append(onehot)
                return self.loss_function(y_true_onehot, y_pred)
        else:
            return self.loss_function(y_true, y_pred)
    
    def evaluate(self, X, y):
        """
        Evalúa el modelo en un conjunto de datos.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tupla (loss, accuracy)
        """
        # Evaluar con dropout desactivado (training=False)
        predictions = self.forward(X, training=False)
        loss = self.compute_loss(y, predictions)
        accuracy = self.compute_accuracy(y, predictions)
        
        return loss, accuracy
    
    def compute_accuracy(self, y_true, y_pred):
        """
        Calcula la accuracy (precisión) de las predicciones.
        
        Formula: Accuracy = (TP + TN) / Total
        
        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones (probabilidades o one-hot)
            
        Returns:
            Accuracy (valor entre 0 y 1)
        """
        # Determinar si es batch
        is_batch = isinstance(y_pred[0], list) if y_pred else False
        
        if not is_batch:
            # Single sample
            y_pred = [y_pred]
            y_true = [y_true]
        
        correct = 0
        total = len(y_true)
        
        for yt, yp in zip(y_true, y_pred):
            # Convertir predicción a clase
            if isinstance(yp, list):
                # One-hot output: tomar argmax
                pred_class = yp.index(max(yp))
            else:
                # Single value: umbral 0.5
                pred_class = 1 if yp >= 0.5 else 0
            
            # Comparar
            true_class = int(yt)
            if pred_class == true_class:
                correct += 1
        
        return correct / total
    
    def get_parameters_count(self):
        """
        Retorna el número total de parámetros entrenables.
        
        Returns:
            Número total de parámetros
        """
        total = 0
        for layer in self.layers:
            total += layer.get_parameters_count()
        return total
    
    def summary(self):
        """
        Imprime un resumen de la arquitectura de la red.
        """
        print("=" * 80)
        print("🧠 ARQUITECTURA DE LA RED NEURONAL")
        print("=" * 80)
        print()
        
        print(f"{'Capa':<15} {'Entrada':<12} {'Salida':<12} {'Activación':<15} {'Parámetros':<12}")
        print("-" * 80)
        
        for i, layer in enumerate(self.layers):
            capa_name = "Input" if i == 0 else f"Hidden {i}" if i < len(self.layers) - 1 else "Output"
            dropout_str = f" (dropout={layer.dropout_rate:.1f})" if layer.dropout_rate > 0 else ""
            print(f"{capa_name:<15} {layer.n_in:<12} {layer.n_out:<12} "
                  f"{layer.activation_name + dropout_str:<15} {layer.get_parameters_count():<12}")
        
        print("-" * 80)
        print(f"{'TOTAL':<54} {self.get_parameters_count():<12}")
        print()
        
        print(f"Función de pérdida: {self.loss_name}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Optimizer: {self.optimizer.name}")  # BONUS
        
        # Mostrar dropout si está habilitado
        has_dropout = any(layer.dropout_rate > 0 for layer in self.layers[:-1])
        if has_dropout:
            dropout_rates = [layer.dropout_rate for layer in self.layers[:-1] if layer.dropout_rate > 0]
            print(f"Dropout: {dropout_rates[0]:.2f} (capas ocultas)")  # BONUS
        
        print()
        print("=" * 80)
        print()
    
    def __repr__(self):
        """Representación en string de la red."""
        layer_sizes = [self.layers[0].n_in] + [layer.n_out for layer in self.layers]
        arch_str = ' → '.join(str(n) for n in layer_sizes)
        return f"Network({arch_str}, params={self.get_parameters_count()})"


# ============================================================================
# TESTS
# ============================================================================

def test_network():
    """
    Tests básicos para validar la red neuronal.
    """
    print("🧪 Testeando Network...")
    print()
    
    # Crear red pequeña: 4 → 3 → 2
    print("1️⃣  Creando red: 4 → 3 → 2")
    network = Network(loss='binary_crossentropy', learning_rate=0.01, optimizer='sgd')
    network.build([4, 3, 2], ['sigmoid', 'softmax'], seed=42)
    network.summary()
    
    # Test forward single
    print("2️⃣  Forward pass (single sample):")
    x = [1.0, 0.5, -0.5, 2.0]
    output = network.forward(x)
    print(f"   Input:  {x}")
    print(f"   Output: [{', '.join(f'{v:.4f}' for v in output)}]")
    print(f"   Sum:    {sum(output):.6f}")
    print()
    
    # Test forward batch
    print("3️⃣  Forward pass (batch):")
    X = [[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 0.0, -1.0]]
    outputs = network.forward(X)
    print(f"   Batch size: {len(X)}")
    for i, out in enumerate(outputs):
        print(f"   Sample {i}: [{', '.join(f'{v:.4f}' for v in out)}]")
    print()
    
    # Test backward
    print("4️⃣  Backward pass:")
    y = [1, 0]  # Labels
    network.backward(y, outputs)
    print(f"   ✓ Gradientes calculados para {len(network.layers)} capas")
    for i, layer in enumerate(network.layers):
        print(f"   Capa {i}: dW shape = {len(layer.dW)}×{len(layer.dW[0])}, "
              f"db shape = {len(layer.db)}")
    print()
    
    # Test train_step
    print("5️⃣  Train step:")
    loss_before = network.compute_loss(y, outputs)
    print(f"   Loss antes:   {loss_before:.6f}")
    
    loss_after = network.train_step(X, y)
    print(f"   Loss después: {loss_after:.6f}")
    print()
    
    print("✅ Todos los tests completados")
    print()


if __name__ == '__main__':
    test_network()
