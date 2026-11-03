# 🧪 `test_auto.sh` — Guía de ejecución y limpieza

## Índice

- [Propósito](#-propósito)
- [Modos de ejecución](#-modos-de-ejecución)
- [Artefactos](#-artefactos)
- [Pausas y gráficos](#-pausas-y-gráficos)
- [Orden recomendado con y sin tests](#-orden-recomendado-con-y-sin-tests)
- [Elección final de limpieza](#-elección-final-de-limpieza)
- [Diferencias entre clean, fclean y continuar](#-diferencias-entre-clean-fclean-y-continuar)
- [Targets del Makefile](#-targets-del-makefile)
- [Limitaciones](#-limitaciones)

## 🎯 Propósito

`test_auto.sh` valida el flujo observable del Multilayer Perceptron sin modificar
el código fuente ni el dataset original. Ejecuta las pruebas en
`test_auto_artifacts/`, una carpeta ignorada por Git.

La suite comprueba:

1. Entorno y estructura.
2. Sintaxis Python.
3. Imports de frameworks prohibidos mediante `ast`.
4. Dataset: 569 muestras, 30 features, etiquetas `B`/`M` e IDs únicos.
5. División reproducible 80/20.
6. Rechazo de ratios inválidos.
7. Test de integración.
8. Entrenamiento real y guardado del modelo.
9. Esquema de arquitectura y `softmax`.
10. Binary Cross-Entropy con un valor conocido.
11. Predicciones y probabilidades normalizadas.
12. Curvas de loss y accuracy.
13. Smoke tests ejecutables de Adam y Dropout.

Cada sección muestra qué hace, por qué importa, un comando manual equivalente,
el resultado esperado y una referencia dinámica `archivo:línea`.

### Formato de cada prueba

Para evitar mezclar explicaciones con resultados, cada comprobación sigue siempre
el mismo orden:

```text
[NN] Título de la prueba
  Qué hace
  Por qué
  Implementación: archivo:línea
  Repetir manualmente
  Resultado esperado
  ✓/✗/⚠ Resultado real
```

La implementación se localiza dinámicamente durante la ejecución. Si una prueba
no depende de una función concreta, se indica que no existe una referencia
dinámica en lugar de reutilizar accidentalmente la referencia de otra prueba.

## ▶️ Modos de ejecución

### Interactivo — recomendado para aprendizaje

```bash
./test_auto.sh
```

El script:

- pausa entre secciones y espera INTRO;
- muestra explicaciones antes de cada prueba;
- intenta abrir los gráficos cuando se generan;
- explica qué debe observarse en ellos;
- pregunta al final cómo limpiar o conservar los resultados.

### No interactivo — CI, SSH o automatización

```bash
./test_auto.sh --non-interactive
```

No pausa ni solicita respuestas. Genera y conserva los artefactos en
`test_auto_artifacts/`.

### Sin abrir visores gráficos

```bash
./test_auto.sh --no-graphs
```

Los PNG se generan, pero no se intenta abrir ningún visor. Las opciones se pueden
combinar:

```bash
./test_auto.sh --non-interactive --no-graphs
```

## 📁 Artefactos

La carpeta `test_auto_artifacts/` puede contener:

| Archivo | Finalidad |
|---|---|
| `loss.png` | Loss de entrenamiento y validación |
| `accuracy.png` | Accuracy de entrenamiento y validación |
| `model.json` | Modelo generado por el smoke test |
| `history.json` | Historial de entrenamiento |
| `predictions.csv` | Predicciones sobre validación |
| `data_training.csv` | Partición de entrenamiento |
| `data_validation.csv` | Partición de validación |
| `adam.json` | Resultado del smoke test de Adam |
| `dropout.json` | Resultado del smoke test de Dropout |

Estos archivos son regenerables y están excluidos por `.gitignore`.

## 📊 Pausas y gráficos

La suite genera gráficos después de entrenar y antes del informe final:

```text
loss.png
accuracy.png
```

La curva de **loss** debería tender a descender. Si la loss de entrenamiento
continúa bajando mientras la de validación sube, puede existir overfitting.

La curva de **accuracy** debería tender a subir. Una distancia grande y
creciente entre entrenamiento y validación también puede indicar overfitting.

En Linux se intenta usar `xdg-open` o `gio`; en macOS, `open`. En un servidor
sin interfaz gráfica, la suite deja la ruta disponible para abrirla manualmente.

## 🧭 Orden recomendado con y sin tests

Los tests validan el proyecto, pero no son necesarios para ejecutar el flujo
principal. Mantener ambos recorridos separados evita confundir una comprobación
con una fase de producción del modelo.

### Flujo normal, sin tests

```bash
# 1. Dependencias: primera ejecución o después de make fclean
make install

# 2. Exploración opcional
make explore

# 3. División explícita del dataset
make split

# 4. Entrenamiento
make train

# 5. Gráficos
make visualize

# 6. Predicción y métricas
make predict
```

`make train` vuelve a ejecutar `make split` automáticamente, por lo que también
es válido usarlo como primer paso de ejecución. Para repetir todo desde cero:

```bash
make re
```

`make re` equivale a `fclean` seguido del flujo principal completo
(`split → train → visualize → predict`) y no ejecuta tests.

### Validación, separada del flujo normal

```bash
# 1. Sintaxis
make check

# 2. Integración
make test

# 3. Entrenamiento corto
make test-train

# 4. Suite exhaustiva
make test-auto
```

`make test` y `make test-train` preparan por sí mismos los datasets divididos.
La suite `test-auto` es autosuficiente y conserva sus resultados en
`test_auto_artifacts/`.

## 🧹 Elección final de limpieza

En modo interactivo, cuando termina la suite, aparece:

```text
clean    : elimina datasets, historial, imágenes y artefactos de test.
fclean   : hace clean y elimina también modelo y entorno virtual.
continuar: conserva todo para inspección manual.
```

### `clean`

Equivale a:

```bash
make clean
```

Elimina:

- `data_training.csv`;
- `data_validation.csv`;
- historiales;
- métricas;
- imágenes PNG;
- `test_auto_artifacts/`;
- cachés `__pycache__` y `.pyc`.

No elimina:

- `data.csv`;
- código fuente;
- documentación;
- `model_weights.npy`;
- `venv/`.

### `fclean`

Equivale a:

```bash
make fclean
```

Hace todo lo anterior y elimina además:

- `model_weights.npy`;
- cualquier `*.npy`;
- `venv/`.

El resultado es el estado de un clon limpio respecto a artefactos generados,
sin tocar código, documentación ni `data.csv`.

### `continuar`

No elimina nada. Conserva los artefactos para:

```bash
ls -lh test_auto_artifacts/
xdg-open test_auto_artifacts/loss.png
xdg-open test_auto_artifacts/accuracy.png
```

## 🔁 Targets del Makefile

| Target | Acción |
|---|---|
| `make install` | Instala dependencias |
| `make explore` | Explora el dataset |
| `make split` | Genera los datasets de entrenamiento y validación |
| `make train` | Entrena el modelo principal |
| `make visualize` | Genera las curvas desde `training_history.json` |
| `make predict` | Ejecuta predicciones y métricas |
| `make check` | Comprueba la sintaxis Python |
| `make test` | Ejecuta el test de integración |
| `make test-train` | Ejecuta un entrenamiento corto de validación |
| `make test-auto` | Ejecuta esta suite automática completa |
| `make clean` | Elimina artefactos regenerables y resultados de tests |
| `make fclean` | Hace `clean` y elimina modelo, `.npy` y entorno virtual |
| `make re` | Hace `fclean` y reconstruye el flujo normal completo |

`make re` no elimina ni regenera el código fuente ni `data.csv`; reconstruye
todo lo que depende de ellos. Puede tardar más porque entrena el modelo completo
y prepara el entorno virtual necesario para visualizar.

## ⚠️ Limitaciones

Pasar la suite ofrece una señal reproducible y trazable, pero no demuestra
ausencia absoluta de errores. Sigue siendo necesario:

- defender las decisiones matemáticas;
- revisar overfitting;
- probar casos límite;
- inspeccionar el código;
- ejecutar la defensa del subject.

La suite nunca borra `data.csv`, el código fuente ni la documentación.
