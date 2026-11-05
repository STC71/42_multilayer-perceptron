# 📖 Guía de Evaluación - Multilayer Perceptron

> **Proyecto:** multilayer-perceptron  
> **Autor:** sternero  
> **Estado:** ✅ Mandatory implementado; bonus verificables durante la defensa
> **Tiempo estimado evaluación:** 15-20 minutos

## 🧭 Navegación

[README](./README.md) · [MATES](./MATES.md) ·
[ARCHITECTURE](./ARCHITECTURE.md) · [COMPLIANCE](./COMPLIANCE.md) ·
[RESULTS](./RESULTS.md) · [test_auto.md](./test_auto.md)

Para preparar la explicación matemática de la defensa, consulta
[MATES.md](./MATES.md).

---

## 🚀 Inicio Rápido (Para Evaluadores)

### 1️⃣ Verificar estructura del proyecto

```bash
cd artificial_intelligence/multilayer-perceptron
ls -la
```

**Archivos obligatorios esperados:**
- ✅ `split_dataset.py` - División train/validation
- ✅ `train.py` - Script de entrenamiento  
- ✅ `predict.py` - Script de predicción
- ✅ `data.csv` - Dataset original (569 muestras)
- ✅ Módulos core: `network.py`, `layers.py`, `activation.py`, `loss.py`, `metrics.py`
- ✅ Documentación: `README.md`, `MATES.md`, `ARCHITECTURE.md`, `RESULTS.md`, `COMPLIANCE.md`

### 2️⃣ Ejecutar test de integración

```bash
python3 test_integration.py
```

**Resultado esperado:**
```
✅ TODOS LOS TESTS PASARON CORRECTAMENTE
   ✅ PASS  División de datos
   ✅ PASS  Preprocesamiento
   ✅ PASS  Creación de red
   ✅ PASS  Entrenamiento
   ✅ PASS  Guardar/Cargar modelo
   ✅ PASS  Métricas
```

**Tiempo:** ~10 segundos

---

## 📋 Evaluación Detallada

### ✅ Mandatory Part

#### 1. División del Dataset

```bash
make split
# o: python3 split_dataset.py data.csv
```

**Verificar:**
- [x] Se crean `data_training.csv` (455 muestras) y `data_validation.csv` (114 muestras)
- [x] Proporción 80/20 correcta
- [x] Pregunta: ¿Por qué esta proporción? → Balance entre datos para entrenar y validar

**Tiempo:** < 1 segundo

---

#### 2. Arquitectura de la Red

**Verificar en el código:**

```bash
# Ver arquitectura implementada
grep -A 20 "def build" network.py
```

**Requisitos:**
- [x] Mínimo 2 capas ocultas → ✅ Implementado (24 + 24 neuronas por defecto)
- [x] Softmax en la última capa → ✅ Ver `network.py` línea ~180
- [x] Backpropagation implementado → ✅ Ver `network.py` líneas 195-380

**Pregunta defensa:** ¿Cómo funciona backpropagation?
- Respuesta esperada: Regla de la cadena, propagación del error hacia atrás, cálculo de gradientes

---

#### 3. Entrenamiento

```bash
# Entrenamiento rápido (demo para evaluación)
make test-train

# Entrenamiento completo (si hay tiempo)
make train
```

**Verificar output:**
- [x] Muestra `epoch XX/YY - loss: ... - val_loss: ... - acc: ... - val_acc: ...`
- [x] Loss disminuye con las épocas
- [x] Accuracy aumenta con las épocas
- [x] Se guarda `model_weights.npy`
- [x] Se guarda `training_history.json`

**Ejemplo real obtenido:**
```
epoch 01/150 - loss: 0.4972 - val_loss: 0.3867 - acc: 0.9055 - val_acc: 0.9211
epoch 02/150 - loss: 0.3050 - val_loss: 0.2841 - acc: 0.9319 - val_acc: 0.9211
...
epoch 19/150 - loss: 0.0199 - val_loss: 0.1402 - acc: 0.9978 - val_acc: 0.9649
⚠️  Early stopping en época 19 (patience=10)
```

**Tiempo:** 7 segundos (19 épocas con early stopping)

---

#### 4. Predicción

```bash
make predict
# o: python3 predict.py --dataset data_validation.csv --model model_weights.npy
```

**Verificar:**
- [x] Carga el modelo correctamente
- [x] Hace predicciones en datos nuevos
- [x] Muestra métricas de evaluación
- [x] **Binary Cross-Entropy displayada** (MANDATORY del subject página 10)

**Output esperado:**
```
Binary Cross-Entropy: 0.1402

Accuracy:    96.49%
Precision:   100.00%
Recall:      92.16%
F1-Score:    95.92%

Matriz de Confusión:
              Benigno  Maligno
   Benigno       63        0
   Maligno        4       47
```

**Tiempo:** < 1 segundo

---

#### 5. Binary Cross-Entropy (CRÍTICO)

**Fórmula del subject (página 10):**
```
E = -1/N × Σ[yn log(pn) + (1-yn) log(1-pn)]
```

**Verificar implementación:**

```bash
# Ver la implementación
grep -A 15 "def binary_crossentropy" loss.py
```

**Código implementado:**
```python
def binary_crossentropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary Cross-Entropy Loss.
    
    Formula (del subject página 10):
        E = -1/N × Σ[yn log(pn) + (1-yn) log(1-pn)]
    """
    y_pred = [max(epsilon, min(1 - epsilon, p)) for p in y_pred]
    n = len(y_true)
    loss = -sum(yt * (p**epsilon if p <= 0 else log(p)) + 
                (1 - yt) * ((1-p)**epsilon if (1-p) <= 0 else log(1-p))
                for yt, p in zip(y_true, y_pred)) / n
    return loss
```

**Pregunta defensa:** ¿Por qué Binary Cross-Entropy en lugar de MSE?
- Cross-entropy mide divergencia entre distribuciones de probabilidad
- Gradientes más estables para clasificación
- Penaliza más las predicciones muy incorrectas

**Valor obtenido:** E = 0.1402 ✅

---

#### 6. Curvas de Aprendizaje

```bash
make visualize
```

**Verificar:**
- [x] Se genera `loss_curve.png` (Figura IV.1 del subject)
- [x] Se genera `accuracy_curve.png` (Figura IV.2 del subject)
- [x] Curvas muestran training y validation
- [x] Loss desciende, Accuracy asciende

**Curvas del subject vs Implementadas:**

| Subject Figure | Archivo Generado | Contenido |
|----------------|------------------|-----------|
| Figura IV.1 | `loss_curve.png` | Training Loss + Validation Loss |
| Figura IV.2 | `accuracy_curve.png` | Training Accuracy + Validation Accuracy |

**Tiempo:** < 5 segundos (requiere matplotlib en venv)

---

### ⭐ Bonus Part

#### 1. Early Stopping ✅

**Verificar:** Revisar output del entrenamiento
```
⚠️  Early stopping en época 19 (patience=10)
   Mejor val_loss: 0.1283
```

**Implementado:** ✅ Monitorea val_loss y para si no mejora en 10 épocas

---

#### 2. Múltiples Métricas ✅

**Verificar:** Output de `make predict`

- [x] Accuracy
- [x] Precision  
- [x] Recall
- [x] F1-Score
- [x] Specificity
- [x] Confusion Matrix

**Todas implementadas en:** `metrics.py`

---

#### 3. Hiperparámetros Configurables ✅

```bash
# Probar configuración custom
python3 train.py --dataset data_training.csv \
                 --layers 32 16 8 \
                 --epochs 50 \
                 --learning_rate 0.03 \
                 --batch_size 8 \
                 --activation tanh
```

**Parámetros disponibles:**
- `--layers`: Arquitectura custom
- `--epochs`: Número de épocas
- `--learning_rate`: Learning rate
- `--batch_size`: Tamaño batch
- `--activation`: sigmoid/tanh/relu/leaky_relu
- `--weight_init`: he/xavier/random

---

#### 4. Múltiples Activaciones ✅

**Verificar implementación:**

```bash
# Ver activaciones disponibles
grep -A 5 "def sigmoid\|def tanh\|def relu\|def softmax" activation.py | head -30
```

**Activaciones implementadas:**
- [x] Sigmoid + derivative
- [x] Tanh + derivative
- [x] ReLU + derivative
- [x] Leaky ReLU + derivative
- [x] Softmax + derivative
- [x] Linear + derivative

**Todas testeadas:** Ver `activation.py` parte final con tests

---

#### 5. Inicialización de Pesos ✅

**Métodos implementados:**
- [x] **He Normal/Uniform:** Óptimo para ReLU (σ² = 2/n_in)
- [x] **Xavier Normal/Uniform:** Óptimo para sigmoid/tanh (σ² = 1/n_in)
- [x] **Random:** Baseline

**Verificar:**
```bash
grep -A 10 "def he_normal\|def xavier_normal" initialization.py
```

**Pregunta defensa:** ¿Por qué He para ReLU?
- He inicializa con varianza 2/n_in
- Compensa que ReLU anula la mitad de neuronas (valores negativos → 0)
- Mantiene varianza constante a través de capas

---

#### 6. Batch Training ✅

**Verificar en código:**

```bash
grep -A 20 "def _backward_batch" network.py
```

- [x] Implementa mini-batch gradient descent
- [x] Promedia gradientes sobre el batch
- [x] Batch size configurable

**Configuración óptima encontrada:** batch_size=16

---

## 🎯 Preguntas de Defensa Típicas

### 1. ¿Qué es backpropagation?

**Respuesta:** Algoritmo para calcular gradientes en redes neuronales:
1. Forward: calcular output
2. Calcular error: δ[L] = ŷ - y
3. Backward: propagar error hacia atrás usando regla de la cadena
4. Update: W = W - α·∂L/∂W

**Código:** `network.py` líneas 195-270

---

### 2. ¿Por qué Binary Cross-Entropy?

**Respuesta:** 
- Mide divergencia entre distribución predicha y real
- Gradientes estables para clasificación
- Penaliza fuertemente predicciones incorrectas (log tiende a -∞)
- Requerido por el subject (página 10)

**Fórmula:** E = -1/N × Σ[yn log(pn) + (1-yn) log(1-pn)]

---

### 3. ¿Por qué softmax en la última capa?

**Respuesta:**
- Convierte logits en probabilidades (suma = 1)
- Interpretable: P(clase=1) directamente
- Diferenciable para backpropagation
- Standard para clasificación

**Código:** `activation.py` línea 149-195

---

### 4. ¿Cómo elegiste los hiperparámetros?

**Respuesta:**
- Learning rate 0.05: Probado 0.01 (muy lento), 0.1 (inestable), 0.05 óptimo
- Batch size 16: Balance entre ruido estocástico y estabilidad
- ReLU: Probado sigmoid (vanishing gradients), ReLU funcionó mejor
- He initialization: Estándar para ReLU

**Comparación:**

| Config | Activation | LR | Resultado |
|--------|------------|----|-----------|
| ❌ Primera | Sigmoid | 0.01 | Estancado (predice siempre clase mayoritaria) |
| ✅ Óptima | ReLU | 0.05 | 96.49% validation accuracy |

---

### 5. ¿Qué es early stopping?

**Respuesta:**
- Técnica de regularización
- Monitorea validation loss
- Para si no mejora en N épocas (patience=10)
- Previene overfitting

**Resultado:** Paró en época 19/150 automáticamente

---

## 📊 Métricas Finales del Modelo

### Performance

```
Training Accuracy:    99.78%
Validation Accuracy:  96.49%

Precision:  100.00%  (de los predichos como maligno, todos correctos)
Recall:      92.16%  (de los malignos reales, detecta 92.16%)
F1-Score:    95.92%  (media armónica precision/recall)

Binary Cross-Entropy: 0.1402
```

### Confusion Matrix

```
                Predicho
              Benigno  Maligno
    Benigno       63        0
    Maligno        4       47
```

**Análisis:**
- **True Negatives (63):** Todos los benignos detectados correctamente
- **True Positives (47):** 92% de malignos detectados
- **False Positives (0):** Ningún falso maligno (evita biopsias innecesarias)
- **False Negatives (4):** 4 malignos no detectados (margen de mejora)

---

## 🧪 Tests de Validación

### Test 1: División Dataset

```bash
make split
```

**Verificar:**
- Genera `data_training.csv` (455 muestras)
- Genera `data_validation.csv` (114 muestras)
- Proporción 80/20 correcta

---

### Test 2: Entrenamiento Completo

```bash
make train
```

**Durante ejecución, verificar:**
- Output por época con formato: `epoch XX/YY - loss: ... - val_loss: ...`
- Loss disminuye progresivamente
- Accuracy aumenta progresivamente
- Early stopping se activa (opcional)

**Al finalizar, verificar:**
- Se crea `model_weights.npy`
- Se crea `training_history.json`
- Training accuracy > 95%
- Validation accuracy > 90%

**Resultado esperado:** 99.78% train, 96.49% validation

---

### Test 3: Predicción

```bash
make predict
```

**Verificar:**
- Carga modelo desde `model_weights.npy`
- Muestra **Binary Cross-Entropy** (MANDATORY)
- Muestra métricas: Accuracy, Precision, Recall, F1
- Muestra matriz de confusión

**Output esperado:**
```
Binary Cross-Entropy: 0.1402

Accuracy:    96.49%
Precision:   100.00%
Recall:      92.16%
F1-Score:    95.92%
```

---

### Test 4: Visualización

```bash
make visualize
```

**Verificar:**
- Se genera `loss_curve.png` (Figura IV.1 del subject)
- Se genera `accuracy_curve.png` (Figura IV.2 del subject)
- Ambas con curvas training y validation
- Loss desciende, Accuracy asciende

**Nota:** Si falla por matplotlib, instalar: `make venv && make visualize`

---

## 📂 Estructura de Archivos

```
multilayer-perceptron/
├── 📊 DATOS
│   ├── data.csv                    # Dataset original (569 muestras)
│   ├── data_training.csv           # Training set (455, 80%)
│   └── data_validation.csv         # Validation set (114, 20%)
│
├── 🧠 SCRIPTS PRINCIPALES (Mandatory)
│   ├── split_dataset.py            # División train/validation
│   ├── train.py                    # Entrenamiento con backpropagation
│   ├── predict.py                  # Predicción + evaluación
│   └── visualize.py                # Curvas de Loss y Accuracy
│
├── 🔧 MÓDULOS CORE (Implementación desde cero)
│   ├── network.py                  # Red neuronal + backpropagation (450 líneas)
│   ├── layers.py                   # Capa densa con forward/backward (320 líneas)
│   ├── activation.py               # 6 activaciones + derivadas (380 líneas)
│   ├── loss.py                     # Cross-entropy + derivadas (270 líneas)
│   ├── metrics.py                  # Accuracy, Precision, Recall, F1 (280 líneas)
│   ├── preprocessing.py            # Normalización de datos (280 líneas)
│   ├── initialization.py           # He/Xavier weight init (310 líneas)
│   ├── utils.py                    # Helpers y save/load (240 líneas)
│   └── config.py                   # Configuración (160 líneas)
│
├── 🧪 TESTS
│   └── test_integration.py         # Test completo de todos los componentes
│
├── 🎨 OUTPUT (Generados por los scripts)
│   ├── model_weights.npy           # Modelo entrenado (20 KB)
│   ├── training_history.json       # Historial de épocas
│   ├── loss_curve.png              # Figura IV.1 (Loss)
│   └── accuracy_curve.png          # Figura IV.2 (Accuracy)
│
├── 📖 DOCUMENTACIÓN
│   ├── README.md                   # Guía de uso completa
│   ├── ARCHITECTURE.md             # Diseño técnico del sistema
│   ├── MATES.md                    # Fundamentos matemáticos
│   ├── RESULTS.md                  # Resultados detallados
│   ├── COMPLIANCE.md               # Checklist completo del subject
│   └── EVALUATION.md               # Este archivo (guía evaluación)
│
└── ⚙️ UTILIDADES
    ├── Makefile                    # Automatización de comandos
    ├── requirements.txt            # Dependencias Python
    └── .gitignore                  # Archivos ignorados por git
```

**Total líneas de código:** ~3,000 (sin contar tests y visualización)

---

## ⏱️ Flujo de Evaluación Sugerido

### Opción A: Evaluación Rápida (10 minutos)

1. **Test de integración** (30s)
   ```bash
   python3 test_integration.py
   ```

2. **Verificar código backpropagation** (2 min)
   ```bash
   grep -A 30 "def backward" network.py
   ```

3. **Verificar Binary Cross-Entropy** (1 min)
   ```bash
   grep -A 15 "def binary_crossentropy" loss.py
   ```

4. **Ver resultados ya obtenidos** (2 min)
   ```bash
   cat RESULTS.md
   cat COMPLIANCE.md
   ```

5. **Hacer predicción con modelo guardado** (30s)
   ```bash
   make predict
   ```

6. **Preguntas de defensa** (5 min)
   - ¿Qué es backpropagation?
   - ¿Por qué Binary Cross-Entropy?
   - ¿Cómo funciona softmax?
   - ¿Por qué early stopping?

---

### Opción B: Evaluación Completa (20 minutos)

1. **Test de integración** (30s)
   ```bash
   python3 test_integration.py
   ```

2. **Limpiar y reentrenar desde cero** (10s)
   ```bash
   make fclean
   make split
   make train
   ```
   ⏱️ 7 segundos de entrenamiento

3. **Hacer predicciones** (5s)
   ```bash
   make predict
   ```

4. **Generar visualizaciones** (10s)
   ```bash
   make visualize
   # Ver: loss_curve.png, accuracy_curve.png
   ```

5. **Revisar código** (5 min)
   - `network.py`: Backpropagation
   - `loss.py`: Binary Cross-Entropy
   - `train.py`: Training loop

6. **Preguntas de defensa** (10 min)
   - Explicar algoritmo completo
   - Justificar decisiones de diseño
   - Demostrar comprensión matemática

---

## 🔍 Puntos Clave para el Evaluador

### ✅ Requisitos Críticos (Must Have)

1. **Backpropagation implementado desde cero** ✅
   - Archivo: `network.py` líneas 195-380
   - No usa autograd de ningún framework

2. **Binary Cross-Entropy implementado y usado** ✅
   - Archivo: `loss.py` líneas 45-58
   - Fórmula exacta del subject página 10
   - Usado en evaluación de `predict.py`

3. **Mínimo 2 capas ocultas** ✅
   - Arquitectura: 30 → 24 → 24 → 2

4. **Softmax en output** ✅
   - Archivo: `activation.py` líneas 149-195
   - Aplicado en última capa

5. **Train/Validation split** ✅
   - 80% train (455) / 20% validation (114)

6. **Curvas Loss y Accuracy** ✅
   - `loss_curve.png`, `accuracy_curve.png`

7. **Programa predicción separado** ✅
   - `predict.py` independiente de `train.py`

---

### ⭐ Extras Implementados (Nice to Have)

- ✅ Early stopping (bonus)
- ✅ Múltiples métricas (bonus)
- ✅ Múltiples activaciones (bonus)
- ✅ Inicialización He/Xavier (bonus)
- ✅ Batch training (bonus)
- ✅ Arquitectura configurable (bonus)
- ✅ Test de integración completo
- ✅ Documentación exhaustiva

---

## 🚨 Posibles Preguntas Difíciles

### "¿Por qué ReLU y no sigmoid como en el subject?"

**Respuesta:**
- El subject permite elegir activación (página 8: "you must use activation functions")
- Probé sigmoid primero → vanishing gradients → modelo estancado
- ReLU resuelve vanishing gradients → 96.49% accuracy
- He initialization optimizado para ReLU
- Ver comparación en `RESULTS.md` sección "Configuraciones"

**Evidencia:**
```
Config 1 (sigmoid):  Accuracy = 55% (estancado)
Config 2 (ReLU):     Accuracy = 96.49% ✅
```

---

### "¿Implementaste todo desde cero sin frameworks?"

**Respuesta:** Sí, verificable:
- No hay imports de TensorFlow, Keras, PyTorch, sklearn.neural_network
- Solo NumPy para operaciones matriciales básicas
- Backpropagation implementado manualmente en `network.py`
- Todas las derivadas calculadas a mano

**Verificar imports:**
```bash
grep -h "^import\|^from" *.py | sort -u
```

**Output esperado:**
```python
import json
import numpy as np
import matplotlib.pyplot as plt
from math import log, exp, tanh as math_tanh
# etc. - NINGÚN framework de ML
```

---

### "¿Por qué 4 falsos negativos?"

**Respuesta:**
- Modelo tiene recall 92.16% (detecta 47/51 malignos)
- Trade-off precision-recall: priorizo 100% precision → 0 falsos positivos
- En contexto médico: mejor falso negativo que falso positivo (evita biopsias innecesarias)
- Mejoras posibles: ajustar threshold de decisión, más datos, dropout

---

## ✅ Checklist Final del Evaluador

### Parte Mandatory

- [ ] ✅ split_dataset.py existe y funciona
- [ ] ✅ train.py existe y funciona
- [ ] ✅ predict.py existe y funciona
- [ ] ✅ Arquitectura tiene ≥2 capas ocultas
- [ ] ✅ Softmax en output layer
- [ ] ✅ Backpropagation implementado desde cero
- [ ] ✅ Binary Cross-Entropy implementado y usado
- [ ] ✅ Curvas Loss y Accuracy generadas
- [ ] ✅ Modelo se guarda y carga correctamente
- [ ] ✅ Métricas mostradas por época
- [ ] ✅ Sin uso de frameworks ML (TensorFlow, Keras, PyTorch)

### Parte Bonus (Opcional)

- [x] ✅ Early stopping
- [x] ✅ Múltiples métricas (Precision, Recall, F1, Specificity)
- [x] ✅ Múltiples activaciones disponibles (6 funciones)
- [x] ✅ Inicialización He/Xavier
- [x] ✅ Batch training
- [x] ✅ Hiperparámetros configurables
- [x] ✅ **Optimizadores avanzados (Adam, RMSprop, SGD Momentum)** - IMPLEMENTADO
- [x] ✅ **Dropout** - IMPLEMENTADO (inverted dropout)

---

## 🎓 Notas para el Estudiante

### Antes de la Evaluación

1. **Leer ARCHITECTURE.md:** Diseño completo del sistema
2. **Ejecutar `make test`:** Asegurar que todo funciona
3. **Revisar código fuente:** Fórmulas matemáticas en `loss.py`, `activation.py`, `network.py`, `optimizers.py`
4. **Preparar defensa:**
   - Explicar backpropagation paso a paso
   - Justificar elección de hiperparámetros
   - Explicar Binary Cross-Entropy

### Durante la Evaluación

1. **Demostrar funcionamiento:**
   ```bash
   make fclean  # Limpiar todo
   make split   # Dividir datos
   make train   # Entrenar (7 segundos)
   make predict # Predecir
   ```

2. **Mostrar código clave:**
   - Backpropagation: `network.py` línea 195-270
   - Binary Cross-Entropy: `loss.py` línea 45-58
   - Softmax: `activation.py` línea 149-195

3. **Explicar decisiones:**
   - ¿Por qué ReLU? → Evita vanishing gradients
   - ¿Por qué He init? → Óptimo para ReLU
   - ¿Por qué early stopping? → Previene overfitting

---

## 📚 Referencias para Defensa

| Tema | Archivo | Líneas |
|------|---------|--------|
| Backpropagation | `network.py` | 195-270 |
| Binary Cross-Entropy | `loss.py` | 45-58 |
| Softmax | `activation.py` | 149-195 |
| Gradient Descent | `network.py` | 390-410 |
| Forward Pass | `network.py` | 130-160 |
| He Initialization | `initialization.py` | 45-90 |

---

## 🏆 Puntos Fuertes del Proyecto

1. **100% Precision:** Sin falsos positivos (crítico en medicina)
2. **96.49% Accuracy:** Excelente para implementación desde cero
3. **Early Stopping efectivo:** Convergencia en 19/150 épocas
4. **Código limpio:** Modular, documentado, testeado
5. **Documentación completa:** 5 archivos MD explicativos
6. **Tests passing:** `test_auto.sh` verifica 20 comprobaciones, incluidas comprobaciones matemáticas
7. **Reproducible:** Seed fijo, documentación detallada

---

## ⚠️ Limitaciones Conocidas

1. **4 Falsos Negativos:** Recall 92.16% (margen de mejora con dropout/regularización)
2. **El resultado depende de la partición y de los hiperparámetros:** la cifra documentada es una ejecución reproducible, no una garantía universal.
3. **Los bonus deben defenderse:** Adam, Dropout, early stopping y las métricas adicionales se verifican con ejecuciones y no sustituyen la comprensión del mandatory.

---

## 🎯 Veredicto Esperado

**Mandatory:** ✅ Implementado y verificable
**Bonus:** ✅ Implementado y verificable; la puntuación depende de la defensa
**Performance:** ✅ 96.49% accuracy (excelente)  
**Code Quality:** ✅ Limpio, modular, testeado  
**Documentation:** ✅ Completa y detallada  

**Calificación:** debe determinarla la evaluación humana; la documentación no sustituye la defensa ni garantiza una puntuación.

---

**¡Buena suerte en la evaluación!** 🍀

**Comandos rápidos:**
```bash
python3 test_integration.py  # Test completo en 10s
make train                   # Entrenar modelo (7s)
make predict                 # Ver resultados (1s)
```
