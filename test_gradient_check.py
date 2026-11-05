#!/usr/bin/env python3
"""Compara gradientes analiticos con diferencias finitas en una red pequena."""

from network import Network


network = Network(loss="categorical_crossentropy", learning_rate=0.01)
network.build([2, 3, 2], ["tanh", "softmax"], weight_init="xavier", seed=7)
inputs = [[0.2, -0.4]]
targets = [1]
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

maximum_error = max(errors)
if maximum_error >= 1e-5:
    raise SystemExit(f"error maximo demasiado alto: {maximum_error}")
print(f"error maximo absoluto: {maximum_error:.3e}")
