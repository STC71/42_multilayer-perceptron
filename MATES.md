# 🧮 Matemáticas del Multilayer Perceptron

> **Guía rigurosa y visual de las matemáticas que utiliza este proyecto**

Este documento explica, desde la intuición hasta las ecuaciones, cómo el
proyecto transforma datos de células en probabilidades, calcula el error y
modifica sus parámetros para aprender. Las fórmulas se han contrastado con
`activation.py`, `loss.py`, `layers.py`, `network.py`, `preprocessing.py`,
`optimizers.py` y `metrics.py`.

> **Importante:** los ejemplos numéricos de esta guía son pequeños y
> didácticos. No son resultados médicos ni sustituyen la evaluación del
> modelo sobre el dataset completo.

## 📚 Índice

- [Mapa de documentación](#-mapa-de-documentación)
- [1. El problema y la notación](#1-el-problema-y-la-notación)
- [2. De una característica a una neurona](#2-de-una-característica-a-una-neurona)
- [3. Feedforward capa a capa](#3-feedforward-capa-a-capa)
- [4. Funciones de activación](#4-funciones-de-activación)
- [5. Preprocesamiento y normalización](#5-preprocesamiento-y-normalización)
- [6. Funciones de pérdida](#6-funciones-de-pérdida)
- [7. Backpropagation](#7-backpropagation)
- [8. Gradient descent y mini-batches](#8-gradient-descent-y-mini-batches)
- [9. Optimizadores bonus](#9-optimizadores-bonus)
- [10. Inicialización de pesos](#10-inicialización-de-pesos)
- [11. Dropout](#11-dropout)
- [12. Early stopping](#12-early-stopping)
- [13. Métricas y matriz de confusión](#13-métricas-y-matriz-de-confusión)
- [14. Estabilidad numérica](#14-estabilidad-numérica)
- [15. Ejemplo completo reducido](#15-ejemplo-completo-reducido)
- [16. Cómo comprobar las matemáticas](#16-cómo-comprobar-las-matemáticas)
- [Glosario](#-glosario)

---

## 🧭 Mapa de documentación

| Documento | Pregunta que responde |
|---|---|
| [README.md](./README.md) | ¿Qué es el proyecto y cómo se ejecuta? |
| **MATES.md** | ¿Qué matemáticas hay detrás de cada paso? |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | ¿Cómo se organizan los módulos y la red? |
| [COMPLIANCE.md](./COMPLIANCE.md) | ¿Qué requisitos del subject se cumplen? |
| [EVALUATION.md](./EVALUATION.md) | ¿Cómo preparar una defensa o evaluación? |
| [RESULTS.md](./RESULTS.md) | ¿Qué resultados se han observado en ejecuciones? |
| [test_auto.md](./test_auto.md) | ¿Cómo se comprueba automáticamente el proyecto? |
| [test_eval.sh](./test_eval.sh) | ¿Cómo se guía una evaluación según la hoja oficial? |

Volver al [índice](#-índice).

## 1. El problema y la notación

El dataset contiene una muestra por paciente, una etiqueta y 30
características numéricas. En el código:

- `B` (benigno) se codifica como `0`.
- `M` (maligno) se codifica como `1`.
- La red produce dos probabilidades: `[p(B), p(M)]`.

Usaremos:

| Símbolo | Significado |
|---|---|
| $x$ | vector de entrada de una muestra |
| $x_j$ | característica número $j$ |
| $W^{(l)}$ | matriz de pesos de la capa $l$ |
| $b^{(l)}$ | vector de bias de la capa $l$ |
| $z^{(l)}$ | valor antes de activar |
| $a^{(l)}$ | salida de la capa, después de activar |
| $y$ | etiqueta real |
| $\hat y$ | predicción de la red |
| $\alpha$ | learning rate |
| $L$ | función de pérdida |

Una neurona es como una pequeña báscula: asigna una importancia a cada
entrada, suma esas contribuciones y aplica una regla que decide cuánto deja
pasar. Una red combina muchas básculas en capas.

## 2. De una característica a una neurona

Para una neurona con entradas $x_1,\ldots,x_n$, pesos
$w_1,\ldots,w_n$ y bias $b$:

$$
z = \sum_{j=1}^{n} w_j x_j + b
$$

En forma vectorial:

$$
z = w^T x + b
$$

El bias permite desplazar la decisión aunque todas las entradas sean cero.
Después se aplica una activación:

$$
a = f(z)
$$

### Ejemplo cotidiano

Imaginemos una decisión de llevar paraguas. La neurona recibe:

- $x_1$: probabilidad de lluvia;
- $x_2$: presencia de nubes;
- $x_3$: viento.

Los pesos expresan cuánto influye cada señal. El bias representa una
predisposición inicial. La suma no es todavía una decisión; es la evidencia
acumulada $z$. La activación convierte esa evidencia en una respuesta útil.

## 3. Feedforward capa a capa

La arquitectura utilizada en la configuración principal es:

```text
30 entradas → 24 ReLU → 24 ReLU → 2 Softmax
```

Para cada capa:

$$
z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = f^{(l)}(z^{(l)})
$$

En la primera capa, $a^{(0)}=x$. La salida de una capa se convierte en la
entrada de la siguiente. En el ejemplo principal hay:

$$
30 \times 24 + 24 = 744
$$

$$
24 \times 24 + 24 = 600
$$

$$
24 \times 2 + 2 = 50
$$

Por tanto, el modelo tiene **1.394 parámetros entrenables** en esa
configuración: pesos y bias. Esta cuenta es una comprobación útil para detectar
arquitecturas mal construidas.

El feedforward es parecido a una cadena de filtros: cada capa resume la
información recibida y entrega una representación más útil a la siguiente.

## 4. Funciones de activación

Las activaciones introducen no linealidad. Sin ellas, varias capas lineales
seguirían siendo equivalente a una sola transformación lineal.

### 4.1 Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Su derivada puede calcularse usando su propia salida:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

Su rango es $(0,1)$. Es intuitiva para una probabilidad binaria, pero puede
tener gradientes muy pequeños cuando $z$ es muy grande o muy pequeño.

### 4.2 Tanh

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

$$
\tanh'(z)=1-\tanh^2(z)
$$

Su salida está centrada en cero, en $(-1,1)$. En el código la derivada
recibe la salida activada, igual que sigmoid.

### 4.3 ReLU

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

$$
\operatorname{ReLU}'(z)=
\begin{cases}
1 & z>0\\
0 & z\leq 0
\end{cases}
$$

Es barata y mantiene un gradiente sencillo en la zona positiva. En $z=0$
la implementación toma la derivada como cero.

### 4.4 Leaky ReLU

$$
\operatorname{LeakyReLU}(z)=
\begin{cases}
z & z>0\\
\lambda z & z\leq 0
\end{cases}
$$

$$
\operatorname{LeakyReLU}'(z)=
\begin{cases}
1 & z>0\\
\lambda & z\leq 0
\end{cases}
$$

Con $\lambda=0.01$, una neurona negativa no queda completamente
inactiva. Es una capacidad bonus disponible en `activation.py`.

### 4.5 Softmax

Para logits $z_1,\ldots,z_K$:

$$
\operatorname{softmax}(z_i)=
\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

Cada salida está entre 0 y 1 y:

$$
\sum_{i=1}^{K}\hat y_i=1
$$

Así, para dos clases, `[0.8, 0.2]` puede leerse como 80 % benigno y
20 % maligno, siempre entendiendo que es una probabilidad producida por el
modelo, no una certeza clínica.

## 5. Preprocesamiento y normalización

Las 30 características pueden estar en escalas muy distintas. Sin normalizar,
una variable con valores grandes dominaría las sumas y los gradientes.

### 5.1 Estandarización Z-score

El proyecto calcula en entrenamiento:

$$
\mu_j=\frac{1}{N}\sum_{i=1}^{N}x_{ij}
$$

$$
\sigma_j=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_j)^2}
$$

Y transforma:

$$
x'_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
$$

Si una característica tiene desviación cero, el código usa `1.0` para evitar
división por cero. La transformación de validación reutiliza las estadísticas
del entrenamiento; no vuelve a calcularlas con los datos de validación.

### 5.2 Min-Max

La alternativa implementada es:

$$
x'=\frac{x-\min(x)}{\max(x)-\min(x)}
$$

Su rango habitual es $[0,1]$. Si máximo y mínimo coinciden, el código usa
`0.0`.

### Ejemplo cotidiano

Comparar una edad entre 0 y 100 con un salario entre 0 y 100.000 sin
normalización sería como pesar dos señales con reglas distintas. La
estandarización pone ambas en una escala comparable alrededor de cero.

## 6. Funciones de pérdida

La pérdida cuantifica qué lejos está la predicción de la respuesta correcta.
El optimizador no “sabe” si una predicción parece razonable: solo intenta
reducir este número.

### 6.1 Categorical Cross-Entropy

Para una muestra one-hot:

$$
L=-\sum_{k=1}^{K}y_k\log(\hat y_k)
$$

Para $N$ muestras:

$$
L_{\text{medio}}=-\frac{1}{N}
\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}\log(\hat y_{ik})
$$

Como solo una componente de $y$ vale 1, únicamente importa el logaritmo de
la probabilidad asignada a la clase correcta. Si la red da 0.9 a la clase
correcta, la penalización es pequeña; si da 0.01, la penalización es grande.

Con softmax, la derivada respecto a los logits se simplifica elegantemente:

$$
\frac{\partial L}{\partial z_k}=\hat y_k-y_k
$$

Esa es la ruta utilizada por `network.py` para la salida softmax.

### 6.2 Binary Cross-Entropy

Es la fórmula exigida por el subject para una probabilidad positiva $p$:

$$
E=-\frac{1}{N}\sum_{n=1}^{N}
\left[y_n\log(p_n)+(1-y_n)\log(1-p_n)\right]
$$

Para $y=1$, importa $-\log(p)$. Para $y=0$, importa
$-\log(1-p)$.

Ejemplo con dos casos:

$$
y=[1,0],\quad p=[0.8,0.3]
$$

$$
E=-\frac{\log(0.8)+\log(0.7)}{2}
\approx 0.2899092476
$$

El código limita $p$ a $[\varepsilon,1-\varepsilon]$, con
$\varepsilon=10^{-10}$, para que `log(0)` no produzca un error.

### 6.3 MSE

La pérdida cuadrática, disponible para regresión, es:

$$
\operatorname{MSE}=\frac{1}{N}\sum_{n=1}^{N}(y_n-\hat y_n)^2
$$

$$
\frac{\partial\operatorname{MSE}}{\partial\hat y_n}
=\frac{2}{N}(\hat y_n-y_n)
$$

No es la pérdida principal de esta clasificación.

## 7. Backpropagation

Backpropagation aplica la regla de la cadena para averiguar cuánto contribuyó
cada parámetro al error final.

### 7.1 Error en la salida

Para softmax + categorical cross-entropy:

$$
\delta^{(L)}=\hat y-y
$$

Este vector indica la dirección en la que cada logit debería moverse.

### 7.2 Propagación hacia una capa anterior

Para una capa oculta:

$$
\delta^{(l)}=
\left((W^{(l+1)})^T\delta^{(l+1)}\right)
\odot f'(z^{(l)})
$$

El símbolo $\odot$ significa producto elemento a elemento. Primero se
reparte el error a través de los pesos de la siguiente capa; después se
aplica la derivada de la activación local.

### 7.3 Gradientes de pesos y bias

Si la entrada de la capa es $a^{(l-1)}$:

$$
\frac{\partial L}{\partial W^{(l)}}=
\delta^{(l)}(a^{(l-1)})^T
$$

$$
\frac{\partial L}{\partial b^{(l)}}=\delta^{(l)}
$$

Cada peso recibe el error de su neurona multiplicado por la activación que
entró por ese peso. Es la razón por la que `layers.py` conserva entradas,
preactivaciones, salidas y deltas.

### 7.4 Analogía

Si una cadena de montaje entrega un producto defectuoso, backpropagation
recorre la cadena hacia atrás y estima cuánto contribuyó cada máquina. No
culpa a todas por igual: usa la regla de la cadena para asignar responsabilidad.

## 8. Gradient descent y mini-batches

Con los gradientes calculados, SGD actualiza:

$$
W\leftarrow W-\alpha\frac{\partial L}{\partial W}
$$

$$
b\leftarrow b-\alpha\frac{\partial L}{\partial b}
$$

El signo menos mueve los parámetros en la dirección que reduce la pérdida.
El learning rate controla el tamaño del paso: demasiado grande puede saltar
por encima del mínimo; demasiado pequeño puede aprender lentamente.

Para un mini-batch de tamaño $m$, el proyecto promedia:

$$
\nabla W_{\text{batch}}=\frac{1}{m}
\sum_{i=1}^{m}\nabla W_i
$$

$$
\nabla b_{\text{batch}}=\frac{1}{m}
\sum_{i=1}^{m}\nabla b_i
$$

Un batch es como pedir opinión a varias muestras antes de corregir la regla,
en vez de reaccionar exageradamente ante un solo caso.

## 9. Optimizadores bonus

### 9.1 SGD con momentum

$$
v_t=\beta v_{t-1}+g_t
$$

$$
\theta_t=\theta_{t-1}-\alpha v_t
$$

La velocidad acumulada suaviza zigzags y puede acelerar una dirección
persistente.

### 9.2 RMSprop

$$
s_t=\beta s_{t-1}+(1-\beta)g_t^2
$$

$$
\theta_t=\theta_{t-1}-
\alpha\frac{g_t}{\sqrt{s_t}+\varepsilon}
$$

Cada parámetro obtiene una escala adaptada a la magnitud reciente de su
gradiente.

### 9.3 Adam

Adam combina primer y segundo momento:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Como ambos empiezan en cero, se corrige el sesgo:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Y se actualiza:

$$
\theta_t=\theta_{t-1}-
\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
$$

La suite comprueba Adam con un entrenamiento real; la explicación oral debe
justificar por qué sus hiperparámetros no tienen por qué coincidir con SGD.

## 10. Inicialización de pesos

Los pesos no deben comenzar todos iguales: si lo hicieran, neuronas de una
misma capa recibirían la misma señal y aprenderían lo mismo.

Para una capa con $n_{\text{in}}$ entradas:

### He

Pensada para ReLU:

$$
\operatorname{Var}(W)\approx\frac{2}{n_{\text{in}}}
$$

### Xavier/Glorot

Adecuada para sigmoid/tanh:

$$
\operatorname{Var}(W)\approx\frac{1}{n_{\text{in}}}
$$

Las variantes uniformes ajustan el intervalo para conservar una varianza
similar. El objetivo es evitar que las activaciones o gradientes crezcan o se
apaguen demasiado al atravesar capas.

## 11. Dropout

Con tasa $r$, cada activación oculta recibe una máscara aleatoria:

$$
m_j\sim\operatorname{Bernoulli}(1-r)
$$

El proyecto usa inverted dropout:

$$
\tilde a_j=a_j\cdot m_j\cdot\frac{1}{1-r}
$$

La esperanza se conserva:

$$
E[\tilde a_j]=a_j(1-r)\frac{1}{1-r}=a_j
$$

Durante inferencia no se aplica máscara ni escalado. Durante backpropagation
el gradiente debe recibir la misma máscara y el mismo factor:

$$
\tilde\delta_j=\delta_j\cdot m_j\cdot\frac{1}{1-r}
$$

Si una neurona fue descartada, su gradiente debe ser cero. Por eso
`test_dropout_gradient.py` no se limita a entrenar: inspecciona esa propiedad.

## 12. Early stopping

El entrenamiento monitoriza `val_loss`. Se conserva el estado con menor
pérdida de validación:

$$
t^*=\arg\min_t L_{\text{validation}}^{(t)}
$$

Si no hay mejora durante `patience` épocas, se detiene el proceso y se
restauran los pesos de $t^*$, no los de la última época.

Es como estudiar hasta que los simulacros dejan de mejorar y conservar la
mejor versión del examen, en vez de quedarse con el último intento aunque sea
peor.

## 13. Métricas y matriz de confusión

Para clasificación binaria:

| | Predicho 0 | Predicho 1 |
|---|---:|---:|
| Real 0 | TN | FP |
| Real 1 | FN | TP |

$$
\operatorname{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\operatorname{Precision}=\frac{TP}{TP+FP}
$$

$$
\operatorname{Recall}=\frac{TP}{TP+FN}
$$

$$
\operatorname{Specificity}=\frac{TN}{TN+FP}
$$

$$
F1=2\frac{\operatorname{Precision}\cdot\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}}
$$

En un contexto de diagnóstico, un falso negativo puede ser especialmente
importante: es un caso maligno que el modelo predice como benigno. Por eso no
conviene interpretar accuracy sin mirar también recall y la matriz completa.

## 14. Estabilidad numérica

Las fórmulas ideales pueden fallar con números finitos:

- `exp(1000)` desborda;
- `log(0)` no está definido;
- dividir por una desviación cero es inválido.

El código responde con:

1. **Softmax estable:** resta el máximo logit:
   $$
   \operatorname{softmax}(z)=\operatorname{softmax}(z-c)
   $$
   para cualquier constante $c$, normalmente $c=\max(z)$.
2. **Clipping:** limita probabilidades BCE antes de aplicar logaritmos.
3. **Desviación de respaldo:** usa `1.0` si una feature es constante.
4. **Épsilon en optimizadores:** evita denominadores exactamente cero.

Estas operaciones no cambian la intención matemática; evitan que una
representación finita destruya el cálculo.

## 15. Ejemplo completo reducido

Consideremos una red mínima con dos entradas, una neurona oculta y dos salidas:

```text
[x1, x2] → [ReLU] → [Softmax: benigno, maligno]
```

1. La capa oculta calcula:
   $$
   z_h=w_{h1}x_1+w_{h2}x_2+b_h
   $$
2. Aplica:
   $$
   a_h=\max(0,z_h)
   $$
3. Cada salida calcula un logit:
   $$
   z_k=w_{kh}a_h+b_k
   $$
4. Softmax convierte los logits en $\hat y_B,\hat y_M$.
5. Si la etiqueta es maligna, $y=[0,1]$, la salida:
   $$
   \hat y=[0.7,0.3]
   $$
   tiene error:
   $$
   \delta_{\text{salida}}=[0.7,-0.7]
   $$
6. Ese error se propaga a la neurona oculta, se calculan los gradientes y se
   actualizan pesos y bias con el optimizador elegido.

No hace falta memorizar cada número: hay que poder explicar el recorrido,
identificar qué representa cada símbolo y justificar el signo de la
actualización.

## 16. Cómo comprobar las matemáticas

Desde el directorio del proyecto:

```bash
# Gradient check de backpropagation
python3 test_gradient_check.py
# Resultado esperado: error maximo absoluto menor que 1e-5

# Propiedad matemática de Dropout
python3 test_dropout_gradient.py
# Resultado esperado: mascara y gradiente de Dropout verificados

# Suite completa, sin pausas ni visor gráfico
./test_auto.sh --non-interactive --no-graphs
```

El gradient check aproxima una derivada con diferencias finitas:

$$
\frac{\partial L}{\partial\theta}
\approx
\frac{L(\theta+\varepsilon)-L(\theta-\varepsilon)}{2\varepsilon}
$$

Después compara esa aproximación con el gradiente analítico de
backpropagation. Un error pequeño no demuestra ausencia absoluta de bugs, pero
es una evidencia fuerte para esa red y esos parámetros probados.

Para ver el orden pedagógico completo, consulta
[test_auto.md](./test_auto.md). Para relacionar cada fórmula con el diseño,
consulta [ARCHITECTURE.md](./ARCHITECTURE.md), y para preparar la explicación
oral, [EVALUATION.md](./EVALUATION.md).

## 📖 Glosario

- **Activación:** función no lineal aplicada a $z$.
- **Bias:** término independiente que desplaza la suma.
- **Batch:** conjunto de muestras procesadas antes de actualizar parámetros.
- **Época:** recorrido completo por el conjunto de entrenamiento.
- **Gradiente:** vector de derivadas que indica cómo cambia la pérdida.
- **Logit:** valor $z$ antes de softmax.
- **Loss:** medida numérica del error.
- **Parámetro:** peso o bias aprendido.
- **Softmax:** transformación de logits en probabilidades que suman uno.

Volver al [índice](#-índice) · Ir al [README principal](./README.md).
