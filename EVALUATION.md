# 📖 Guía de Evaluación y Defensa - Multilayer Perceptron

> **Proyecto:** multilayer-perceptron  
> **Autor:** sternero  
> **Estado:** ✅ Mandatory y Bonus implementados y verificables  
> **Tiempo estimado evaluación:** 15-20 minutos

---

## 🧭 Navegación

[README](./README.md) · [MATES](./MATES.md) ·
[ARCHITECTURE](./ARCHITECTURE.md) · [COMPLIANCE](./COMPLIANCE.md) ·
[RESULTS](./RESULTS.md) · [test_auto.md](./test_auto.md)

* Para una evaluación guiada alineada con la hoja oficial (`scale`), usa [`test_eval.sh`](./test_eval.sh).
* Para la suite exhaustiva automatizada, usa [`test_auto.sh`](./test_auto.sh).
* Para revisar las derivadas y ecuaciones paso a paso, consulta [MATES.md](./MATES.md).

---

## 🚀 Inicio Rápido (Para Evaluadores)

### 1️⃣ Verificar estructura e independencia de frameworks

```bash
cd artificial_intelligence/multilayer-perceptron
ls -la
grep -h "^import\|^from" *.py | sort -u
```

**Verificación clave:** No existen imports de TensorFlow, Keras, PyTorch ni `sklearn.neural_network`. Todo el motor matemático está programado desde cero usando únicamente operaciones matriciales básicas con NumPy.

**Archivos obligatorios esperados:**
* ✅ `split_dataset.py` — División train/validation
* ✅ `train.py` — Script de entrenamiento con backpropagation
* ✅ `predict.py` — Script de predicción y cálculo de Binary Cross-Entropy
* ✅ `visualize.py` — Generación de curvas de aprendizaje
* ✅ `data.csv` — Dataset original (569 muestras, 30 características + diagnóstico)
* ✅ Módulos core: `network.py`, `layers.py`, `activation.py`, `loss.py`, `metrics.py`, `optimizers.py`, `preprocessing.py`, `initialization.py`

### 2️⃣ Ejecutar test de integración

```bash
python3 split_dataset.py data.csv
python3 test_integration.py
```

**Resultado esperado (~10 segundos):**
```
✅ TODOS LOS TESTS PASARON CORRECTAMENTE
   ✅ PASS  División de datos
   ✅ PASS  Preprocesamiento
   ✅ PASS  Creación de red
   ✅ PASS  Entrenamiento
   ✅ PASS  Guardar/Cargar modelo
   ✅ PASS  Métricas
```

---

## 📋 Protocolo Oficial de Evaluación (Rúbrica 42)

### 1. Implementation & Training (Parte Obligatoria)

#### A. División del Dataset (si no se ha hecho antes) y Preprocesamiento
```bash
make split
# Equivalente a: python3 split_dataset.py data.csv
```
* **Qué comprobar:** Se generan `data_training.csv` (80% - 455 muestras) y `data_validation.csv` (20% - 114 muestras).
* **Preprocesamiento:** Los parámetros de escalado (media y desviación estándar) se calculan **exclusivamente** sobre el conjunto de entrenamiento y se guardan junto al modelo para aplicarlos después sobre los datos de prueba sin contaminar el experimento (*data leakage*).

#### B. Arquitectura y Softmax
```bash
grep -A 20 "def build" network.py
```
* **Mínimo 2 capas ocultas:** Cumplido por defecto (profundidad mínima de 4 capas contando entrada y salida) y configurable mediante `--layers`.
* **Softmax en la salida:** Implementado en `activation.py` (líneas 149-195) para devolver una distribución de probabilidades cuya suma es 1.

#### C. Entrenamiento y Guardado
```bash
make train
```
* **Métricas por época:** Muestra `loss`, `val_loss`, `acc` y `val_acc` en cada época.
* **Guardado:** Al terminar guarda la topología, los pesos y los parámetros de normalización en `model_weights.npy`, además del historial en `training_history.json`.

#### D. Curvas de Aprendizaje
```bash
make visualize
```
* Genera `loss_curve.png` (Figura IV.1 del subject) y `accuracy_curve.png` (Figura IV.2 del subject), mostrando tanto la curva de entrenamiento como la de validación.

**📖 Cómo leer e interpretar estas dos gráficas (Explicación sencilla):**

* **Gráfica 1: `loss_curve.png` (La curva de los fallos — cuanto más abajo, mejor):**
  * **Qué muestra:** La línea azul (*Training Loss*) mide el error de la red mientras estudia sus apuntes; la línea roja (*Validation Loss*) mide el error al enfrentarse a un examen sorpresa con casos nuevos.
  * **Qué vemos aquí:** Ambas líneas caen en picado desde la época 1 hasta estabilizarse cerca de `0.11` (validación) y `0.02` (entrenamiento). Alrededor de la época 15, la línea roja toca su punto más bajo (`0.1154`) y a partir de ahí empieza a subir muy suavemente mientras la azul sigue bajando: ahí es justo donde actúa nuestro **Early Stopping**, deteniendo el proceso y guardando los pesos de la época 15 antes de que la red empiece a memorizar en lugar de aprender.
* **Gráfica 2: `accuracy_curve.png` (La curva de los aciertos — cuanto más arriba, mejor):**
  * **Qué muestra:** El porcentaje de pacientes diagnosticados correctamente en cada época (donde `1.00` = 100% de aciertos).
  * **Qué vemos aquí:** En solo 3 épocas el modelo salta del `92%` a superar el `95.6%` en casos nuevos (línea roja), alcanzando el **`96.49%`** en la época 15 y manteniéndose estable sin caídas bruscas.
* **Analogía cotidiana (El estudiante preparando un examen):**
  * Imagina que preparas un examen haciendo tests de práctica en casa (**línea azul**) y cada tarde haces un simulacro oficial con preguntas nuevas (**línea roja**). Los primeros días mejoras muchísimo en ambos; pero si sigues estudiando exactamente los mismos tests de casa durante semanas, sacarás un 10 en casa (`99.78%`) pero dejarás de mejorar en preguntas nuevas (`96.49%`). El punto óptimo para presentarte al examen es justo cuando tu nota en los simulacros nuevos toca techo (época 15).

---

### 2. Prediction & Model Performances (Prueba en Vivo con `evaluation.py`)

Esta sección corresponde exactamente al bloque **Prediction** de la hoja de corrección oficial.

#### Paso 1: Generar los datos oficiales de evaluación y dividir `data_training.csv` (se ejecuta UNA sola vez)
> **💡 Paso imprescindible:** `evaluation.py` crea `data_training.csv` (427 muestras) y `data_test.csv` (142 muestras). Como `train.py` necesita un archivo `data_validation.csv` para monitorizar el aprendizaje y el *Early Stopping*, dividimos el `data_training.csv` oficial (en ~341 train y ~86 validation), dejando `data_test.csv` **100% intacto y virgen** para el examen final.

```bash
python3 evaluation.py
python3 split_dataset.py data_training.csv
```
> **⚠️ Importante:** **No vuelvas a ejecutar ninguno de estos dos comandos entre intento e intento**, para que los 3 intentos se entrenen y evalúen sobre exactamente los mismos archivos (`data_training.csv`, `data_validation.csv` y `data_test.csv`).

---

#### Paso 2: Entrenar y evaluar hasta 3 modelos (quedándose con el mejor)

La rúbrica oficial permite entrenar **hasta 3 modelos** sobre `data_training.csv` (ya que el entrenamiento depende de factores aleatorios como la inicialización de pesos o el dropout) y quedarse con el **valor más bajo** obtenido en `data_test.csv`.

Los 3 intentos usan **`Tanh + Xavier + Adam`**, que es la combinación que mejor calibra las probabilidades para bajar la *Binary Cross-Entropy* por debajo de **`0.08`** (5/5 puntos):

* **🥇 Intento 1 — Configuración Principal (`Tanh + Xavier`, `seed 42`, `dropout 0.3`):**
  ```bash
  ./train.py --dataset data_training.csv --layers 24 12 --activation tanh --weight_init xavier --optimizer adam --learning_rate 0.002 --dropout 0.3 --epochs 100 --batch_size 16 --early_stopping --patience 15 --seed 42
  python3 predict.py --dataset data_test.csv --model model_weights.npy
  ```

* **🥈 Intento 2 — Variación de Semilla y Regularización (`Tanh + Xavier`, `seed 7`, `dropout 0.25`):**
  ```bash
  ./train.py --dataset data_training.csv --layers 24 12 --activation tanh --weight_init xavier --optimizer adam --learning_rate 0.002 --dropout 0.25 --epochs 100 --batch_size 16 --early_stopping --patience 15 --seed 7
  python3 predict.py --dataset data_test.csv --model model_weights.npy
  ```

* **🥉 Intento 3 — Red Compacta con Más Paciencia (`Tanh + Xavier`, `layers 16 8`, `seed 21`, `dropout 0.2`):**
  ```bash
  ./train.py --dataset data_training.csv --layers 16 8 --activation tanh --weight_init xavier --optimizer adam --learning_rate 0.002 --dropout 0.2 --epochs 120 --batch_size 16 --early_stopping --patience 20 --seed 21
  python3 predict.py --dataset data_test.csv --model model_weights.npy
  ```

---

#### Paso 3: ¿Qué número hay que mirar y dónde aparece?

Es fundamental distinguir la salida de cada script para no confundirse durante la corrección:

1. **En `./train.py ...` NO sale la nota:** Los valores `loss` y `val_loss` que aparecen época a época solo sirven para monitorizar el aprendizaje interno.
2. **La nota final sale exclusivamente en `python3 predict.py ...`:** Al ejecutar `predict.py` sobre `data_test.csv`, busca el bloque **`📏 Evaluación con Binary Cross-Entropy (según subject):`** en la terminal:

```text
📏 Evaluación con Binary Cross-Entropy (según subject):

   Binary Cross-Entropy: 0.067881   <-- ESTE ES EL ÚNICO NÚMERO QUE MIRA EL EVALUADOR
```

Se anota el **valor más bajo** de `Binary Cross-Entropy` conseguido entre los 3 intentos y se busca en la siguiente tabla:

---

#### Paso 4: Tabla oficial de puntuación (Binary Cross-Entropy en `data_test.csv`)

| Valor de Binary Cross-Entropy ($E$) | Puntuación | Estado |
| :--- | :---: | :--- |
| Mayor a `0.35` | **0 / 5** | Insuficiente |
| Entre `0.35` y `0.25` | **1 / 5** | Básico |
| Entre `0.25` y `0.18` | **2 / 5** | Aceptable |
| Entre `0.18` y `0.13` | **3 / 5** | Bueno |
| Entre `0.13` y `0.08` | **4 / 5** | Muy bueno |
| **Menor a `0.08`** | **5 / 5** | **Excelente (Objetivo cumplido — `< 0.08`)** |

---

## ⭐ Bonus Part (Verificables durante la defensa)

* **Optimizadores avanzados (`optimizers.py`):** Implementación desde cero de `sgd_momentum`, `rmsprop` y `adam`.
* **Early Stopping (`network.py`):** Detiene el entrenamiento si `val_loss` deja de mejorar durante `N` épocas (`--patience`) y restaura los mejores pesos.
* **Dropout (`layers.py`):** Regularización mediante *inverted dropout* (`--dropout`) para evitar el sobreajuste.
* **Inicialización de pesos (`initialization.py`):** Métodos `he`, `xavier` y `random`.
* **Funciones de activación (`activation.py`):** 6 funciones con sus derivadas (`sigmoid`, `tanh`, `relu`, `leaky_relu`, `softmax`, `linear`).
* **Métricas adicionales (`metrics.py`):** Accuracy, Precision, Recall, F1-Score, Specificity y Matriz de Confusión.
* **Historial y gráficas combinadas (`visualize.py`):** Exportación a JSON y visualización comparativa.

---

## 💡 Glosario Didáctico (Conceptos Clave Explicados de Forma Sencilla)

Esta sección sirve como apoyo rápido durante la defensa para explicar cualquier término técnico de forma clara y con analogías cotidianas.

### 1. Datos y Preparación

* **Dataset:** Es la tabla completa de datos con la que trabaja el proyecto. En nuestro caso son **569 casos médicos** (filas), cada uno con **30 mediciones** de células (columnas) y el diagnóstico real (`M` = Maligno, `B` = Benigno).
* **Media ($\mu$):** El promedio de una columna de datos (sumar todos los valores y dividir entre el total).
* **Desviación estándar ($\sigma$):** Mide qué tan dispersos o separados están los datos respecto a la media. *Ejemplo:* Si en una clase todos miden 1.75 m, la desviación es casi 0; si hay niños de 1.20 m y adultos de 1.95 m, la desviación es alta.
* **Normalización (Min-Max):** Escalar todos los datos para que quepan exactamente entre **0 y 1**. *Ejemplo:* Convertir notas de distintos exámenes (unos sobre 10, otros sobre 100) a una escala común de 0 a 1.
* **Estandarización (Z-score):** Transformar los datos para que tengan **media 0 y desviación estándar 1** usando $z = \frac{x - \mu}{\sigma}$. Es crucial aquí porque una columna puede medir el área del tumor (valores de `500` a `2500`) y otra la suavidad (valores de `0.05` a `0.15`); sin estandarizar, la red creería que el área es más importante solo por tener números más grandes.
* **Seed (Semilla aleatoria):** Un número fijo que controla el generador de números aleatorios. Sirve para que, si repetimos el experimento hoy o mañana, los datos se mezclen igual y los pesos empiecen con los mismos valores, haciendo el resultado reproducible.

### 2. Estructura de la Red Neuronal

* **Neurona (Perceptrón):** La unidad básica de cálculo. Recibe varios números de entrada, los multiplica por su importancia (**pesos**), suma un ajuste base (**bias**) y pasa el resultado por una **función de activación**.
* **Capas (*Layers*):** Grupos de neuronas organizadas en fila:
  * *Capa de entrada (Input):* Recibe los 30 datos del paciente.
  * *Capas ocultas (Hidden):* Capas intermedias donde la red combina datos y detecta patrones complejos (por ejemplo, combinaciones de tamaño + irregularidad).
  * *Capa de salida (Output):* Entrega el veredicto final (2 neuronas: probabilidad de Benigno y probabilidad de Maligno).

### 3. Funciones de Activación (Deciden cuánto "se enciende" una neurona)

* **Sigmoid:** Aplasta cualquier número para dejarlo entre **0 y 1** en forma de curva "S". Fue muy usada históricamente, pero en redes profundas hace que el aprendizaje se vuelva muy lento (*vanishing gradient*), como se ve si entrenamos solo 100 épocas con SGD básico.
* **Tanh (Tangente hiperbólica):** Aplasta los valores entre **-1 y 1**, centrando las salidas alrededor de 0. En nuestro dataset médico ofrece una calibración de probabilidades superior a ReLU, evitando predicciones excesivamente extremas y logrando bajar la *Binary Cross-Entropy* por debajo de `0.07`.
* **ReLU (*Rectified Linear Unit*):** Si el número es negativo lo convierte en **0**, y si es positivo lo deja pasar tal cual ($\max(0, x)$). *Analogía:* Es como un filtro que ignora señales negativas/irrelevantes y deja pasar sin freno las señales útiles. Permite entrenar muy rápido, aunque puede generar salidas grandes que hacen que `softmax` sea demasiado confiado ante un error.
* **Softmax:** Se usa exclusivamente en la capa final. Convierte los números crudos de salida en **porcentajes que suman 100% (1.0)**. *Ejemplo:* Transforma `[2.0, 0.5]` en `[0.82, 0.18]` (82% Benigno, 18% Maligno).

### 4. Aprendizaje y Optimización

* **Época (*Epoch*):** Una vuelta completa en la que la red neuronal ha visto y estudiado **todos** los ejemplos del conjunto de entrenamiento una vez. *Analogía:* Leer todo el temario de un examen de principio a fin una vez; entrenar 30 épocas es darle 30 repasos completos al temario.
* **Función de pérdida (*Loss function*):** La fórmula matemática que mide **qué tan equivocada está la red**. Cuanto más cerca de **0**, mejor. Aquí usamos **Binary Cross-Entropy**, que castiga muy severamente cuando la red dice estar "99.9% segura" de un diagnóstico que resulta ser falso.
* **Loss vs. Val_Loss:**
  * `loss`: El error de la red sobre los datos que está estudiando (entrenamiento).
  * `val_loss`: El error sobre datos nuevos que **no** usa para estudiar (validación). Es el indicador real de si está aprendiendo o simplemente memorizando.
* **Acc / Val_Acc (*Accuracy*):** El porcentaje de aciertos directos (ej. `0.9859` = 98.59% de pacientes diagnosticados correctamente).
* **Learning Rate (Tasa de aprendizaje):** El tamaño de los pasos que da la red al corregir sus errores.
  * *Muy bajo (ej. `0.0001`):* Avanza como una hormiga; necesita miles de épocas.
  * *Muy alto (ej. `1.0`):* Da saltos gigantes y se pasa de largo del punto óptimo sin lograr estabilizarse.
* **SGD (*Stochastic Gradient Descent*):** El método clásico para ajustar los pesos. Calcula la pendiente del error y da un paso fijo hacia abajo.
* **Adam (*Adaptive Moment Estimation*):** Un optimizador inteligente moderno. Combina **inercia** (si lleva varias épocas bajando en la misma dirección, acelera) con **pasos adaptativos** (ajusta la velocidad individualmente para cada parámetro). Por eso converge mucho más rápido que SGD.

### 5. Problemas Comunes y Soluciones

* **Overfitting (Sobreajuste):** Ocurre cuando la red **memoriza** los ejemplos de entrenamiento en lugar de aprender reglas generales. *Analogía:* Un estudiante que se aprende de memoria las respuestas de un examen de prueba: saca un 10 en ese examen (`loss` casi 0), pero suspende el examen real con preguntas nuevas (`val_loss` empieza a subir).
* **Dropout:** Técnica de regularización que "apaga" aleatoriamente un porcentaje de neuronas (ej. 30% con `--dropout 0.3`) en cada paso del entrenamiento. *Analogía:* Obligar a un equipo a trabajar cada día sin algunos de sus miembros para que nadie dependa de una sola persona y todos aprendan a resolver el problema de forma robusta.
* **Early Stopping (Parada temprana):** Un mecanismo de seguridad que vigila `val_loss`. Si ve que el error en validación deja de mejorar durante varias épocas seguidas (ej. `patience=15`), detiene el entrenamiento y recupera los pesos del mejor momento antes de que empezara el *overfitting*.

---

## 🎯 Las 4 Preguntas Obligatorias de la Defensa (Guía Rápida)

La rúbrica oficial evalúa de 0 a 5 la comprensión de estos 4 conceptos:

1. **Feedforward (Propagación hacia adelante):**
   * Los datos entran por la primera capa y avanzan capa por capa hasta la salida.
   * En cada capa se realiza una multiplicación matricial ($Z = X \cdot W + b$) y se aplica una función de activación ($A = f(Z)$).
   * Al final, Softmax entrega las probabilidades de cada clase.

2. **Gradient Descent (Descenso del gradiente):**
   * Algoritmo de optimización cuyo objetivo es encontrar los pesos $W$ y sesgos $b$ que minimizan la función de pérdida.
   * Calcula la derivada (gradiente) del error respecto a cada peso para saber hacia dónde sube el error, y actualiza el peso en la dirección opuesta proporcionalmente al *learning rate* ($W_{nuevo} = W_{actual} - \alpha \cdot \nabla W$).

3. **Backpropagation (Retropropagación):**
   * Es el algoritmo que permite calcular esos gradientes de forma eficiente desde la última capa hacia la primera utilizando la **regla de la cadena** del cálculo diferencial.
   * Primero calcula cuánto se equivocó la salida ($\delta^{[L]} = \hat{y} - y$) y reparte la culpa hacia atrás capa por capa para saber cuánto contribuyó cada peso al error final.

4. **Overfitting (Sobreajuste):**
   * Sucede cuando el modelo se ajusta demasiado al ruido o particularidades del set de entrenamiento, perdiendo capacidad de generalizar en datos nuevos.
   * **Cómo se detecta en nuestras gráficas:** `loss` sigue bajando hacia 0 mientras `val_loss` se estanca o empieza a subir.
   * **Cómo lo prevenimos en este proyecto:** Con la división *train/validation*, *Early Stopping* y regularización mediante *Dropout*.

---

## 💡 Argumentos Clave si el Evaluador Pregunta por el Diseño

### ¿Por qué `Tanh + Xavier + Dropout 0.3` funciona mejor aquí que `ReLU`?
* **Calibración de probabilidades (clave para *Binary Cross-Entropy*):** En datasets tabulares médicos pequeños (569 muestras), `ReLU` puede generar activaciones lineales grandes que vuelven a `softmax` excesivamente confiado (ej. `0.999999` vs `0.000001`). Como la entropía cruzada penaliza logarítmicamente ($-\log(p)$), un solo fallo con confianza extrema dispara el error global. `Tanh` acota las salidas entre `-1` y `1`, manteniendo probabilidades suaves y bien calibradas (`0.0678` de *Binary Cross-Entropy* y **98.59% de Accuracy**).
* **Inicialización Xavier:** Está diseñada matemáticamente para mantener estable la varianza de las activaciones y gradientes en funciones simétricas como `tanh`.
* **Dropout (`0.3`) + Early Stopping:** Evita que los 1070 parámetros de la red memoricen las 427 muestras de entrenamiento, logrando **0 falsos positivos** (`100%` Precision y Specificity) en datos completamente desconocidos (`data_test.csv`).

---

## 📁 Mapa Rápido del Código para Mostrar al Evaluador

| Concepto | Archivo | Líneas aprox. |
| :--- | :--- | :---: |
| **Forward Pass** | `network.py` | `130 - 160` |
| **Backpropagation** | `network.py` | `195 - 270` |
| **Gradient Descent / Update** | `network.py` / `optimizers.py` | `390 - 410` |
| **Binary Cross-Entropy** | `loss.py` | `45 - 58` |
| **Softmax y Activaciones** | `activation.py` | `149 - 195` |
| **Inicialización He / Xavier** | `initialization.py` | `45 - 90` |
