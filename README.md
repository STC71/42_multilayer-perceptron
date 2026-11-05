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
- [Guía detallada de test_auto.sh](#-guía-detallada-de-test_autosh)
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

El flujo recomendado separa claramente la ejecución del proyecto de su
validación. Los tests no son pasos necesarios para entrenar y utilizar el
modelo, aunque sí son muy aconsejables antes de presentarlo o modificarlo.

### 🚀 Flujo normal, sin tests

#### 1. Instalar dependencias

```bash
make install
```

Solo es necesario hacerlo la primera vez o después de una limpieza del entorno.

#### 2. Explorar el dataset (opcional)

```bash
make explore
```

Permite revisar distribución, dimensiones y estadísticas antes de entrenar.

#### 3. Dividir el dataset

```bash
make split
```

También se ejecuta automáticamente como dependencia de `make train`.

```bash
python3 split_dataset.py data.csv
# Genera: data_training.csv y data_validation.csv
```

**¿Qué hace?**
- ✓ Lee el dataset original
- ✓ Mezcla aleatoriamente las muestras
- ✓ Divide en 80% entrenamiento / 20% validación
- ✓ Guarda los datasets separados

#### 4. Entrenar el modelo

```bash
# Configuración óptima (ReLU + learning rate 0.05)
python3 train.py --dataset data_training.csv \
                 --layers 24 24 \
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

### 📊 5. Generar gráficos

Después del entrenamiento:

```bash
make visualize
```

Este target crea `loss_curve.png` y `accuracy_curve.png` a partir de
`training_history.json`.

### 🔮 6. Hacer predicciones

```bash
python3 predict.py --model model_weights.npy --dataset data_validation.csv
# o usando Makefile
make predict
```

### ⚡ Atajo del flujo normal completo

Para limpiar los resultados anteriores y repetir todos los pasos principales:

```bash
make re
```

Este target ejecuta, en orden, `fclean`, división del dataset, entrenamiento,
visualización y predicción. No ejecuta ningún test.

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

## 🧪 Validación separada del flujo normal

Los siguientes targets comprueban el proyecto, pero no forman parte del uso
normal del modelo. Se recomienda ejecutarlos en este orden:

```bash
# 1. Sintaxis Python
make check

# 2. Test de integración
make test

# 3. Entrenamiento corto
make test-train

# 4. Suite completa: estructura, datos, entrenamiento, predicción y gráficos
make test-auto
```

`make test` divide el dataset y ejecuta `test_integration.py`. `make test-train`
también prepara previamente la partición de datos. Por tanto, cada prueba puede
ejecutarse desde un estado limpio sin depender de un comando anterior.

La suite automática es más amplia que los tests individuales y genera sus
resultados aislados en `test_auto_artifacts/`. Su documentación detallada está
en [test_auto.md](./test_auto.md).

`test_auto.sh` ejecuta la validación en `test_auto_artifacts/`, una carpeta ignorada por Git, comprueba los requisitos obligatorios, verifica la reproducibilidad y el rechazo de entradas inválidas en la división 80/20, exige validación durante el entrenamiento, realiza un gradient check numérico, inspecciona el esquema del modelo, valida matemáticamente la Binary Cross-Entropy, comprueba predicciones y probabilidades, genera las curvas con un backend no interactivo, verifica la restauración de early stopping y ejecuta smoke tests reales de Adam y Dropout. Cada acción explica qué hace, por qué importa y localiza dinámicamente su implementación como `archivo:línea`. Los bonus se informan por separado y no pueden ocultar un fallo obligatorio.

La suite está diseñada para ofrecer una señal reproducible y trazable, no para prometer ausencia absoluta de bugs: la defensa, las pruebas exploratorias y la revisión del código siguen siendo necesarias.

La suite crea `test_auto_artifacts/` en el directorio del proyecto. Esta carpeta está ignorada por Git y contiene los artefactos de la ejecución. Los gráficos se intentan abrir inmediatamente después de generarse mediante el visor predeterminado del sistema; en un servidor sin interfaz gráfica se informa de la ruta para abrirlos manualmente.

```bash
./test_auto.sh
```

Durante una ejecución interactiva se puede pulsar INTRO entre secciones. Al final se elige:

- `clean`: elimina datasets, historial, imágenes y artefactos del test.
- `fclean`: hace lo anterior y elimina también modelo y entorno virtual.
- `continuar`: conserva los artefactos para inspección manual.

Para automatización o CI:

```bash
./test_auto.sh --non-interactive
```

En ese modo no hay pausas ni visor gráfico interactivo; los ficheros quedan en `test_auto_artifacts/`. `--no-graphs` evita incluso el intento de abrir el visor, pero conserva los PNG.

## 📈 Generación manual de gráficos

Sí, el proyecto debe generar gráficos: el subject exige curvas de `Loss` y `Accuracy` para entrenamiento y validación. Sirven para observar si el modelo aprende, si la pérdida disminuye, si la precisión mejora y si aparece overfitting (por ejemplo, training mejora mientras validation empeora).

Flujo manual completo:

```bash
# 1. Crear los conjuntos de datos
python3 split_dataset.py data.csv

# 2. Entrenar y guardar el historial
python3 train.py \
  --dataset data_training.csv \
  --validation data_validation.csv \
  --layers 24 24 \
  --epochs 150 \
  --learning_rate 0.05 \
  --batch_size 16 \
  --activation relu \
  --weight_init he \
  --early_stopping \
  --save model_weights.npy \
  --history training_history.json

# 3. Generar las imágenes
MPLBACKEND=Agg python3 visualize.py \
  --history training_history.json \
  --loss_output loss_curve.png \
  --accuracy_output accuracy_curve.png

# 4. Comprobar que existen
ls -lh loss_curve.png accuracy_curve.png
```

Los archivos se crean en el directorio actual. `make visualize` también los genera, pero requiere que exista previamente `training_history.json`; `make clean` elimina los PNG y `make fclean` elimina además el modelo y el entorno virtual.

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
├── test_auto.sh                 # Suite automática completa
├── test_auto.md                 # Guía de la suite automática
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
- [Guía completa de test_auto.sh](./test_auto.md)

## 🏆 Autor

**sternero** · 42 Málaga · Branch Artificial Intelligence · [GitHub @STC71](https://github.com/STC71)

---

<div align="center">

[⬅️ Volver a Artificial Intelligence](https://github.com/STC71/42_Outer_Core/tree/main/artificial_intelligence)

[⬆️ Volver arriba](#-multilayer-perceptron)

</div>
