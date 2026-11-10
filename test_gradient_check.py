#!/usr/bin/env python3
"""Compara gradientes analíticos con diferencias finitas en una red pequeña.

El gradiente analítico lo calcula backpropagation. El gradiente numérico
aproxima la misma derivada modificando ligeramente cada peso y bias.
"""

from network import Network


network = Network(loss="categorical_crossentropy", learning_rate=0.01)
network.build([2, 3, 2], ["tanh", "softmax"], weight_init="xavier", seed=7)
inputs = [[0.2, -0.4], [-0.3, 0.6]]
targets = [1, 0]
predictions = network.forward(inputs)
network.backward(targets, predictions)
errors = []
epsilon = 1e-6

for layer in network.layers:
    for row_index in range(layer.n_out):
        for column_index in range(layer.n_in):
            original = layer.weights[row_index][column_index]
            layer.weights[row_index][column_index] = original + epsilon
            loss_plus = network.compute_loss(targets, network.forward(inputs))
            layer.weights[row_index][column_index] = original - epsilon
            loss_minus = network.compute_loss(targets, network.forward(inputs))
            layer.weights[row_index][column_index] = original
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            errors.append(abs(numerical - layer.dW[row_index][column_index]))
    for row_index in range(layer.n_out):
        original = layer.bias[row_index]
        layer.bias[row_index] = original + epsilon
        loss_plus = network.compute_loss(targets, network.forward(inputs))
        layer.bias[row_index] = original - epsilon
        loss_minus = network.compute_loss(targets, network.forward(inputs))
        layer.bias[row_index] = original
        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        errors.append(abs(numerical - layer.db[row_index]))

maximum_error = max(errors)
tolerance = 1e-5
print("🔬 Comprobación de gradientes")
print("   La red calcula cómo debe cambiar cada peso mediante")
print("   backpropagation. Comprobamos esos cálculos con una")
print("   aproximación numérica: modificamos ligeramente cada peso")
print("   y observamos cómo cambia la pérdida.")
print()
print(f"   Error máximo absoluto: {maximum_error:.3e}")
print("   Es la mayor diferencia encontrada entre ambos métodos,")
print("   sin tener en cuenta si la diferencia es positiva o negativa.")
print(f"   En decimal: {maximum_error:.13f}")
print(f"   Límite permitido: {tolerance:.5f}")

if maximum_error >= tolerance:
    print()
    print("❌ El error es demasiado alto.")
    print("   Los dos métodos no coinciden suficientemente. Puede haber")
    print("   un error en backpropagation, en una derivada o en el")
    print("   cálculo numérico.")
    raise SystemExit(f"error maximo demasiado alto: {maximum_error}")

print()
print("✅ El error es mucho menor que el límite permitido.")
print("   Los dos métodos coinciden y backpropagation")
print("   parece funcionar correctamente.")
