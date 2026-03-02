# 🧠 Multilayer Perceptron

> **Red neuronal artificial para clasificación binaria**  
> Implementa una red neuronal desde cero para predecir si un cáncer de mama es maligno o benigno.

---

<div align="center">

*"A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**, <br>
if its performance at tasks in **T**, as measured by **P**, improves with experience **E**."*

*"Se dice que un programa informático aprende de la experiencia **E** con respecto a cierta clase de tareas **T** y una medida de rendimiento **P**, 
si su rendimiento en las tareas de **T**, medido por **P**, mejora con la experiencia **E**."*

— **Tom M. Mitchell**

Profesor e investigador en aprendizaje automático en la Universidad Carnegie Mellon (CMU); autor del libro 'Machine Learning'.

</div>

---

## 📋 Descripción

Este proyecto implementa un **Multilayer Perceptron (Perceptrón Multicapa)** desde cero, una red neuronal artificial feedforward para clasificación binaria. El objetivo es diagnosticar casos de cáncer de mama como malignos o benignos basándose en características de células nucleares.

🎯 **Objetivo:** Construir y entrenar una red neuronal profunda usando backpropagation y gradient descent, sin usar librerías de redes neuronales pre-construidas (como TensorFlow, Keras o PyTorch).

## 🎯 Características

<table>
<tr>
<td width="50%">

### ✅ Parte Obligatoria

| Componente | Descripción |
|------------|-------------|
| 🔀 **split_dataset.py** | División datos entrenamiento/validación |
| 🧠 **train.py** | Entrenamiento con backpropagation |
| 🔮 **predict.py** | Predicción con modelo entrenado |
| 🏗️ **Arquitectura** | Mínimo 2 capas ocultas |
| 🎲 **Activación** | Softmax en capa de salida |
| 📈 **Curvas** | Loss y Accuracy (train/validation) |
| 📊 **Métricas** | Binary Cross-Entropy |
| 🧹 **Preprocesamiento** | Normalización de datos |

</td>
<td width="50%">

### ⭐ Parte Bonus

| Componente | Descripción |
|------------|-------------|
| 🎛️ **Modularidad** | Configuración flexible de capas |
| ⚙️ **Hiperparámetros** | Learning rate, batch size, epochs |
| 🎨 **Visualización** | Gráficos interactivos de métricas |
| 🔧 **Inicializadores** | He/Xavier para pesos |
| 📐 **Activaciones** | Múltiples funciones (sigmoid, tanh, ReLU) |
| 🔄 **Early Stopping** | Detección automática de convergencia |

</td>
</tr>
</table>

## 📊 Dataset

El dataset **Breast Cancer Wisconsin (Diagnostic)** contiene diagnósticos de cáncer de mama:

| Campo | Descripción | Valores |
|-------|-------------|---------|
| **Diagnosis** | Etiqueta de clasificación | M (Maligno) / B (Benigno) |
| **Features** | 30 características de células nucleares | Valores continuos |
| **Samples** | Total de muestras | 569 casos |

📌 **División recomendada:** 80% entrenamiento (455 casos) / 20% validación (114 casos)  
🔬 **Características:** Radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría, dimensión fractal (mean, standard error, worst para cada una)

## 🚀 Instalación

### 📦 Requisitos básicos
```bash
Python 3.x  # Ya instalado en sistemas 42 
```

### 🎨 Para visualizaciones
```bash
pip install matplotlib numpy pandas
# o usar el Makefile
make install
```

## 💻 Uso

### 🔀 1. Dividir el dataset

```bash
python3 split_dataset.py data.csv
# Genera: data_training.csv y data_validation.csv
```

**¿Qué hace?**
- ✓ Lee el dataset original
- ✓ Mezcla aleatoriamente las muestras
- ✓ Divide en 80% entrenamiento / 20% validación
- ✓ Guarda los datasets separados

### 🧠 2. Entrenar el modelo

```bash
python3 train.py --dataset data_training.csv --epochs 100
# o usando Makefile
make train
```

**Ejemplo de salida:**
```
x_train shape : (342, 30)
x_valid shape : (85, 30)

epoch 01/100 - loss: 0.6882 - val_loss: 0.6788 - accuracy: 0.5000 - val_accuracy: 0.5294
epoch 02/100 - loss: 0.6234 - val_loss: 0.6105 - accuracy: 0.6549 - val_accuracy: 0.6824
...
epoch 100/100 - loss: 0.0640 - val_loss: 0.0474 - accuracy: 0.9912 - val_accuracy: 0.9882

> Saving model 'saved_model.npy' to disk...
```

**Parámetros configurables:**
- `--epochs`: Número de iteraciones (default: 100)
- `--learning_rate`: Tasa de aprendizaje (default: 0.01)
- `--batch_size`: Tamaño del mini-batch (default: 32)
- `--layers`: Arquitectura de capas ocultas (default: 24 24)

### 🔮 3. Hacer predicciones

```bash
python3 predict.py --model saved_model.npy --dataset data_validation.csv
# o usando Makefile
make predict
```

**¿Qué hace?**
- ✓ Carga el modelo entrenado
- ✓ Procesa el dataset de validación
- ✓ Realiza predicciones
- ✓ Calcula métricas de evaluación

## 🧮 Conceptos Clave

### 🔄 Feedforward
Propagación hacia adelante de los datos desde la capa de entrada hasta la salida:
```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
```

### 🔙 Backpropagation
Propagación hacia atrás del error para actualizar los pesos:
1. Calcular el error en la salida
2. Propagar el error hacia atrás capa por capa
3. Calcular gradientes usando la regla de la cadena
4. Actualizar pesos con gradient descent

### 📉 Gradient Descent
Minimización de la función de pérdida ajustando los pesos iterativamente:
```
w_new = w_old - learning_rate × ∂Loss/∂w
```

### 🎲 Funciones de Activación

| Función | Fórmula | Uso |
|---------|---------|-----|
| **Sigmoid** | σ(x) = 1/(1+e^-x) | Capas ocultas (valores 0-1) |
| **Tanh** | tanh(x) = (e^x - e^-x)/(e^x + e^-x) | Capas ocultas (valores -1 a 1) |
| **ReLU** | ReLU(x) = max(0, x) | Capas ocultas (evita vanishing gradient) |
| **Softmax** | σ(x)ᵢ = e^xᵢ / Σe^xⱼ | Capa de salida (probabilidades) |

## 📈 Visualización

El programa genera automáticamente gráficos de:

### 📊 Learning Curves
- **Loss curve**: Muestra la evolución de la pérdida durante el entrenamiento
- **Accuracy curve**: Muestra la precisión en train y validation

<div align="center">

```
Loss                          Accuracy
0.7│                         1.0│        ┌─── Validation
   │\                           │      ┌─┘
0.5│ \    Training              │    ┌─┘
   │  \                      0.8│  ┌─┘
0.3│   ────┐                    │┌─┘
   │        └──── Validation    │└─── Training
0.1│                         0.6│
   └───────────────            └───────────────
```

</div>

## 🧪 Testing

```bash
# Ejecutar todos los tests
make test

# Test individual del split
python3 test_split.py

# Test de entrenamiento con validación
python3 test_train.py

# Test de predicción
python3 test_predict.py
```

## 📚 Recursos de Aprendizaje

### 📺 Videos Recomendados
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [StatQuest - Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw)
- [Dot CSV - Redes Neuronales](https://www.youtube.com/watch?v=MRIv2IwFTPg)

### 📖 Documentación
- [Neural Networks and Deep Learning (libro online)](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book - Goodfellow](https://www.deeplearningbook.org/)

## 🔧 Estructura del Proyecto

```
multilayer-perceptron/
├── README.md                    # Este archivo
├── data.csv                     # Dataset original
├── split_dataset.py             # División del dataset
├── train.py                     # Programa de entrenamiento
├── predict.py                   # Programa de predicción
├── mlp/
│   ├── __init__.py
│   ├── network.py              # Clase MLP
│   ├── layers.py               # Implementación de capas
│   ├── activations.py          # Funciones de activación
│   ├── losses.py               # Funciones de pérdida
│   ├── optimizers.py           # Algoritmos de optimización
│   └── utils.py                # Utilidades (normalización, etc.)
├── tests/
│   └── test_*.py               # Tests unitarios
└── Makefile                     # Automatización de tareas
```

## ⚖️ Requisitos del Proyecto

✅ Implementación sin librerías de redes neuronales (TensorFlow, Keras, PyTorch)  
✅ Permitido usar librerías para álgebra lineal (NumPy) y visualización (Matplotlib)  
✅ Mínimo 2 capas ocultas en la arquitectura  
✅ Función softmax en la capa de salida  
✅ Backpropagation implementado manualmente  
✅ Gradient descent para optimización  
✅ División del dataset en train/validation  
✅ Curvas de learning (loss y accuracy)  
✅ Evaluación con binary cross-entropy  

## 🎓 Conceptos para la Evaluación

Durante la defensa, deberás explicar:

1. **Feedforward**: Cómo fluyen los datos a través de la red
2. **Backpropagation**: Cómo se calculan y propagan los gradientes
3. **Gradient Descent**: Cómo se actualizan los pesos
4. **Funciones de Activación**: Por qué se usan y cuál elegir
5. **Overfitting**: Cómo detectarlo y prevenirlo
6. **Learning Rate**: Su impacto en el entrenamiento
7. **Batch vs Stochastic**: Diferencias y trade-offs

## 🏆 Autor

Proyecto completado como parte del currículo de **42 School** - Branch Artificial Intelligence

---

<div align="center">

[⬅️ Volver a Artificial Intelligence](../) 

</div>
