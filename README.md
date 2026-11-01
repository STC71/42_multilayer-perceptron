# 🧠 Multilayer Perceptron

> **Red neuronal artificial para clasificación binaria**  
> Implementa una red neuronal desde cero para predecir si un cáncer de mama es maligno o benigno.

## 📚 Índice

- [Resultados](#-resultados-obtenidos)
- [Descripción y objetivos](#-descripción)
- [Características](#-características)
- [Dataset](#-dataset)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Conceptos clave](#-conceptos-clave)
- [Visualización y testing](#-visualización)
- [Estructura](#-estructura-real-del-proyecto)
- [Requisitos y defensa](#️-requisitos-del-proyecto)
- [Documentación relacionada](#-documentación-relacionada)
- [Autor](#-autor)

---

## 🏆 Resultados Obtenidos

| Métrica | Training | Validation |
|---------|----------|------------|
| **Accuracy** | **99.78%** | **96.49%** |
| **Loss** | 0.0199 | 0.1402 |
| **Precision** | - | **100%** |
| **Recall** | - | **92.16%** |
| **F1-Score** | - | **95.92%** |
| **Épocas** | 19 (early stopping) | - |
| **Tiempo** | 7 segundos | - |

🎯 **Matriz de Confusión:**
- True Positives: 47 (malignos detectados)
- True Negatives: 63 (benignos detectados)  
- False Positives: 0 (¡sin falsos malignos!)
- False Negatives: 4 (4 malignos no detectados)

📊 Ver resultados detallados en [RESULTS.md](RESULTS.md)

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
| 🚀 **Adam Optimizer** | Convergencia rápida con momentum adaptativo |
| 🎲 **Dropout** | Regularización con desactivación aleatoria |

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
Python 3.x
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
# Configuración óptima (ReLU + learning rate 0.05)
python3 train.py --dataset data_training.csv \
                 --layers 30 24 24 2 \
                 --epochs 150 \
                 --learning_rate 0.05 \
                 --batch_size 16 \
                 --activation relu \
                 --weight_init he \
                 --early_stopping \
                 --verbose

# O simplemente:
make train
```

**Ejemplo de salida:**
```
x_train shape: (455, 30)
x_valid shape: (114, 30)

epoch 01/150 - loss: 0.4972 - val_loss: 0.3867 - acc: 0.9055 - val_acc: 0.9211
epoch 02/150 - loss: 0.3050 - val_loss: 0.2841 - acc: 0.9319 - val_acc: 0.9211
epoch 03/150 - loss: 0.2059 - val_loss: 0.2013 - acc: 0.9495 - val_acc: 0.9211
...
epoch 19/150 - loss: 0.0199 - val_loss: 0.1402 - acc: 0.9978 - val_acc: 0.9649

⚠️  Early stopping en época 19 (patience=10)
✅ Modelo guardado en 'model_weights.npy'
```

**Parámetros configurables:**
- `--epochs`: Número máximo de iteraciones (default: 100)
- `--learning_rate`: Tasa de aprendizaje (default: 0.01, **óptimo: 0.05 para SGD, 0.005 para Adam**)
- `--batch_size`: Tamaño del mini-batch (default: 32, **óptimo: 16**)
- `--layers`: Arquitectura personalizada (default: 30 24 24 2)
- `--activation`: Función activación (sigmoid/tanh/**relu**/leaky_relu)
- `--weight_init`: He/Xavier/Random (default: auto)
- `--optimizer`: Optimizador (sgd/**adam**/rmsprop/sgd_momentum) **[BONUS]**
- `--dropout`: Tasa dropout 0.0-0.5 (default: 0.0) **[BONUS]**
- `--early_stopping`: Detener si validación no mejora

#### 🎁 Entrenar con Bonus Features

**Ejemplo 1: Adam optimizer (converge más rápido)**
```bash
python3 train.py --dataset data_training.csv \
                 --optimizer adam \
                 --learning_rate 0.005 \
                 --activation relu \
                 --weight_init he \
                 --early_stopping \
                 --verbose
```
**Resultado:** 100% train / 96.49% validation en **14 épocas** (vs 19 con SGD)

**Ejemplo 2: Dropout regularization**
```bash
python3 train.py --dataset data_training.csv \
                 --dropout 0.2 \
                 --learning_rate 0.05 \
                 --activation relu \
                 --weight_init he \
                 --early_stopping \
                 --verbose
```
**Nota:** Dropout no es necesario para este dataset pequeño (569 samples). Early stopping es suficiente.

**Ejemplo 3: Adam + Dropout combinados**
```bash
python3 train.py --optimizer adam \
                 --dropout 0.2 \
                 --learning_rate 0.005 \
                 --batch_size 16 \
                 --verbose
```

**Otros optimizers disponibles:**
```bash
--optimizer sgd              # Gradient Descent básico (mandatory)
--optimizer sgd_momentum     # SGD con momentum (β=0.9)
--optimizer rmsprop          # Adaptive learning rate
--optimizer adam             # State-of-the-art (momentum + RMSprop)
```

### 🔮 3. Hacer predicciones

```bash
python3 predict.py --model model_weights.npy --dataset data_validation.csv
# o usando Makefile
make predict
```

**Ejemplo de salida real:**
```
📊 REPORTE DE EVALUACIÓN
   Accuracy:    96.49%
   Precision:   100.00%
   Recall:      92.16%
   F1-Score:    95.92%

   Matriz de Confusión:
              Benigno  Maligno
   Benigno       63        0
   Maligno        4       47
```

**¿Qué hace?**
- ✓ Carga el modelo entrenado
- ✓ Procesa el dataset de validación
- ✓ Realiza predicciones
- ✓ Calcula métricas de evaluación (Binary Cross-Entropy + métricas)

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

Ejemplo de visualización esperada:

```
    Loss                              Accuracy
    0.7│                             1.0│         
       │\                               │        ╱ ─────
    0.5│ \                              │      ╱  
       │  \      Training            0.8│    ╱    
    0.3│   \                            │  ╱      
       │    \ ────────                  │╱       Training
       │            Validation          │        
    0.1│                             0.6│        Validation
       └──────────────────              └──────────────────
         Epochs →                         Epochs →
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
make test

# Test de integración disponible en el repositorio
python3 test_integration.py
```

`make test` y `test_integration.py` son las rutas de prueba actualmente presentes. No se documentan scripts individuales que no existen en el árbol.

## 📚 Recursos de Aprendizaje

### 📺 Videos Recomendados
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [StatQuest - Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw)
- [Dot CSV - Redes Neuronales](https://www.youtube.com/watch?v=MRIv2IwFTPg)

### 📖 Documentación
- [Neural Networks and Deep Learning (libro online)](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book - Goodfellow](https://www.deeplearningbook.org/)

## 🔧 Estructura real del proyecto

```
multilayer-perceptron/
├── README.md                    # Este archivo
├── ARCHITECTURE.md              # Diseño técnico
├── COMPLIANCE.md                # Requisitos y cumplimiento
├── EVALUATION.md                # Guía de evaluación
├── RESULTS.md                   # Resultados medidos
├── Makefile                     # Automatización de tareas
├── requirements.txt             # Dependencias Python
├── data.csv                     # Dataset original
├── config.py                    # Configuración
├── preprocessing.py             # Preparación y normalización
├── split_dataset.py             # División del dataset
├── train.py                     # Programa de entrenamiento
├── predict.py                   # Programa de predicción
├── explore_dataset.py           # Exploración del dataset
├── visualize.py                 # Curvas y gráficos
├── test_integration.py          # Test de integración
├── network.py                   # Red neuronal
├── layers.py                    # Capas
├── activation.py                # Activaciones
├── loss.py                      # Funciones de pérdida
├── optimizers.py                # Optimizadores
├── initialization.py            # Inicialización de pesos
├── metrics.py                   # Métricas
└── utils.py                     # Utilidades
```

La lista refleja la organización actual del repositorio. Los archivos generados durante el entrenamiento (`data_training.csv`, `data_validation.csv`, `model_weights.npy` y gráficos) no forman parte del código fuente y pueden eliminarse con `make clean`/`make fclean`.

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

## 📖 Documentación relacionada

- [Arquitectura](./ARCHITECTURE.md)
- [Cumplimiento del subject](./COMPLIANCE.md)
- [Guía de evaluación](./EVALUATION.md)
- [Resultados detallados](./RESULTS.md)

## 🏆 Autor

**sternero** · 42 Málaga · Branch Artificial Intelligence · [GitHub @STC71](https://github.com/STC71)

---

<div align="center">

[⬅️ Volver a Artificial Intelligence](https://github.com/STC71/42_Outer_Core/tree/main/artificial_intelligence)

[⬆️ Volver arriba](#-multilayer-perceptron)

</div>
