#!/usr/bin/env python3
""" La línea anterior es para que el script pueda ejecutarse directamente en Linux y macOS.
    Dicha línea se llama shebang y le indica al sistema operativo que use el intérprete 
    de Python 3 para ejecutar este script."""

""" Estre script prueba la implementación de Dropout en la red neuronal.
    Verifica que Dropout anula el gradiente de las neuronas descartadas."""

from network import Network
""" Network es la clase que implementa la red neuronal multicapa (MLP) con Dropout.
    Se importa desde el módulo network.py."""

network = Network(loss="categorical_crossentropy", learning_rate=0.01)
network.build(
    [2, 3, 2],
    ["relu", "softmax"],
    weight_init="he",
    dropout_rate=0.5,
    seed=42,
)
network.train_step([[1.0, 1.0]], [1])
hidden = network.layers[0]
mask = hidden.dropout_mask[0]
for neuron_index, active in enumerate(mask):
    if active == 0.0 and any(
        abs(value) > 1e-12 for value in hidden.dW[neuron_index]
    ):
        raise SystemExit("una neurona descartada conserva gradiente")
if not any(active == 0.0 for active in mask):
    raise SystemExit("la semilla no produjo ninguna neurona descartada")
print("mascara y gradiente de Dropout verificados")
