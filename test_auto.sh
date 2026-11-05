#!/usr/bin/env bash

# ============================================================================
# test_auto.sh - Suite automatica del Multilayer Perceptron
# ============================================================================
# Las pruebas usan una carpeta de artefactos ignorada por Git. El script no
# modifica el codigo fuente y termina con codigo 0 solo si pasa lo obligatorio.
# ============================================================================

set -u
set -o pipefail

ROJO='\033[0;31m'
VERDE='\033[0;32m'
AMARILLO='\033[1;33m'
AZUL='\033[0;34m'
CIRAN='\033[0;36m'
NEGRITA='\033[1m'
RESET='\033[0m'

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$RAIZ/test_auto_artifacts"
LOGS="$TMP/logs"
INTERACTIVO=0
PAUSA=1
ABRIR_GRAFICOS=1
ANCHO_CABECERA=68
if [[ -t 0 && -t 1 ]]; then
    INTERACTIVO=1
fi

for argumento in "$@"; do
    case "$argumento" in
        --no-pause|--non-interactive)
            INTERACTIVO=0
            PAUSA=0
            ;;
        --no-graphs)
            ABRIR_GRAFICOS=0
            ;;
        --help|-h)
            printf '%s\n' \
                "Uso: ./test_auto.sh [--no-pause|--non-interactive] [--no-graphs]" \
                "  --no-pause         Ejecuta sin esperar INTRO entre secciones." \
                "  --no-graphs        Genera los PNG pero no intenta abrirlos."
            exit 0
            ;;
        *)
            printf '%bArgumento desconocido:%b %s\n' "$ROJO" "$RESET" "$argumento" >&2
            exit 2
            ;;
    esac
done

rm -rf "$TMP"
mkdir -p "$LOGS"

TOTAL=0
PASADAS=0
FALLIDAS=0
AVISOS=0
OBLIGATORIAS=0
BONUS_PASADOS=0
DETALLE_QUE=""
DETALLE_POR_QUE=""
DETALLE_IMPLEMENTACION=""

limpiar() {
    :
}
trap limpiar EXIT

cabecera() {
    local titulo="$1"
    local espacios=$((ANCHO_CABECERA - ${#titulo} - 2))
    if [[ "$PAUSA" -eq 1 && "$INTERACTIVO" -eq 1 && "$TOTAL" -gt 0 ]]; then
        printf '\n%bPulsa INTRO para continuar con la siguiente seccion...%b' "$AMARILLO" "$RESET"
        read -r
    fi
    printf '\n%b╔════════════════════════════════════════════════════════════════════╗%b\n' "$AZUL" "$RESET"
    printf '%b║%b %s%*s %b║%b\n' "$AZUL" "$NEGRITA" "$titulo" "$espacios" "" "$AZUL" "$RESET"
    printf '%b╚════════════════════════════════════════════════════════════════════╝%b\n' "$AZUL" "$RESET"
}

abrir_grafico() {
    local archivo="$1"
    printf '  %bGrafico generado:%b %s\n' "$VERDE" "$RESET" "$archivo"
    if [[ "$ABRIR_GRAFICOS" -eq 0 ]]; then
        printf '  %bVisualizacion:%b desactivada mediante --no-graphs\n' "$AMARILLO" "$RESET"
        return
    fi
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$archivo" >/dev/null 2>&1 &
        printf '  %bVisualizacion:%b se ha solicitado al visor predeterminado del sistema\n' "$CIRAN" "$RESET"
    elif command -v gio >/dev/null 2>&1; then
        gio open "$archivo" >/dev/null 2>&1 &
        printf '  %bVisualizacion:%b se ha solicitado a Gio\n' "$CIRAN" "$RESET"
    elif command -v open >/dev/null 2>&1; then
        open "$archivo" >/dev/null 2>&1 &
        printf '  %bVisualizacion:%b se ha solicitado al visor de macOS\n' "$CIRAN" "$RESET"
    else
        printf '  %bVisualizacion:%b no hay visor grafico disponible; abre el archivo manualmente\n' "$AMARILLO" "$RESET"
    fi
}

finalizar() {
    local eleccion
    if [[ "$INTERACTIVO" -eq 0 ]]; then
        printf '\n%bArtefactos conservados en:%b %s\n' "$AMARILLO" "$RESET" "$TMP"
        return
    fi
    printf '\n%bLimpieza final%b\n' "$AZUL" "$RESET"
    printf '  clean   : elimina datasets, historial, imagenes y artefactos de test.\n'
    printf '  fclean  : hace clean y elimina tambien modelo y entorno virtual.\n'
    printf '  continuar: conserva todo para inspeccion manual.\n'
    while true; do
        printf '%bElige [clean/fclean/continuar] (continuar):%b ' "$AMARILLO" "$RESET"
        read -r eleccion
        eleccion="${eleccion:-continuar}"
        case "$eleccion" in
            clean)
                make -C "$RAIZ" clean >/dev/null
                rm -rf "$TMP"
                printf '%bClean ejecutado. Se han eliminado los artefactos generados.%b\n' "$VERDE" "$RESET"
                break
                ;;
            fclean)
                make -C "$RAIZ" fclean >/dev/null
                rm -rf "$TMP"
                printf '%bFclean ejecutado. Tambien se han eliminado modelo y entorno virtual.%b\n' "$VERDE" "$RESET"
                break
                ;;
            continuar)
                printf '%bArtefactos conservados en:%b %s\n' "$AMARILLO" "$RESET" "$TMP"
                break
                ;;
            *)
                printf '%bRespuesta no valida. Escribe clean, fclean o continuar.%b\n' "$ROJO" "$RESET"
                ;;
        esac
    done
}

explicar() {
    DETALLE_QUE="$1"
    DETALLE_POR_QUE="$2"
    DETALLE_IMPLEMENTACION=""
}

guia_manual() {
    printf '  %bQue hace:%b %s\n' "$CIRAN" "$RESET" "$DETALLE_QUE"
    printf '  %bPor que:%b %s\n' "$CIRAN" "$RESET" "$DETALLE_POR_QUE"
    printf '  %bImplementacion:%b %s\n' "$CIRAN" "$RESET" "${DETALLE_IMPLEMENTACION:-no localizada dinamicamente}"
    printf '  %bRepetir manualmente:%b %s\n' "$CIRAN" "$RESET" "$1"
    printf '  %bResultado esperado:%b %s\n' "$CIRAN" "$RESET" "$2"
}

referencia() {
    local archivo="$1"
    local patron="$2"
    local linea
    linea="$(grep -nE "$patron" "$RAIZ/$archivo" 2>/dev/null | head -1 | cut -d: -f1)"
    if [[ -n "$linea" ]]; then
        DETALLE_IMPLEMENTACION="$archivo:$linea"
    else
        DETALLE_IMPLEMENTACION="no localizada dinamicamente ($archivo)"
    fi
}

pasar() {
    PASADAS=$((PASADAS + 1))
    printf '  %b✓%b %s\n' "$VERDE" "$RESET" "$1"
}

fallar() {
    FALLIDAS=$((FALLIDAS + 1))
    printf '  %b✗%b %s\n' "$ROJO" "$RESET" "$1"
}

avisar() {
    AVISOS=$((AVISOS + 1))
    printf '  %b⚠%b %s\n' "$AMARILLO" "$RESET" "$1"
}

prueba() {
    TOTAL=$((TOTAL + 1))
    printf '%b[%02d]%b %s\n' "$CIRAN" "$TOTAL" "$RESET" "$1"
}

obligatoria() {
    OBLIGATORIAS=$((OBLIGATORIAS + 1))
}

ejecutar() {
    local nombre="$1"
    shift
    if "$@" >"$LOGS/prueba-$TOTAL.log" 2>&1; then
        pasar "$nombre"
        return 0
    fi
    fallar "$nombre"
    sed -n '1,35p' "$LOGS/prueba-$TOTAL.log"
    return 1
}

copiar_fuentes() {
    cp "$RAIZ"/*.py "$RAIZ/data.csv" "$TMP/"
}

cabecera "MULTILAYER PERCEPTRON · SUITE AUTOMATICA"
printf '  %bProyecto:%b %s\n' "$CIRAN" "$RESET" "$RAIZ"
printf '  %bArtefactos:%b %s\n' "$CIRAN" "$RESET" "$TMP"
printf '  %bResultado:%b las obligatorias deben pasar; los bonus se informan aparte\n' "$CIRAN" "$RESET"
printf '  %bModo:%b %s\n' "$CIRAN" "$RESET" "$([[ "$INTERACTIVO" -eq 1 ]] && echo interactivo || echo no-interactivo)"

cabecera "1 · ENTORNO Y ESTRUCTURA"
obligatoria
explicar "Comprueba que Python 3 puede ejecutar la suite." \
    "Sin el interprete no se puede evaluar el proyecto."
prueba "Python 3 disponible"
guia_manual "python3 --version" "Se muestra Python 3.x y el comando termina con codigo 0."
if command -v python3 >/dev/null 2>&1; then
    pasar "$(python3 --version)"
else
    fallar "No se encontro python3"
fi

obligatoria
explicar "Busca los puntos de entrada, modulos, dataset y configuracion necesarios." \
    "Evita obtener un falso aprobado ejecutando solo una parte del proyecto."
referencia "train.py" "def main"
prueba "Archivos obligatorios presentes"
guia_manual "find . -maxdepth 1 -type f | sort" "Aparecen data.csv, los modulos Python, Makefile y requirements.txt."
archivos=(
    data.csv split_dataset.py preprocessing.py network.py layers.py
    activation.py loss.py metrics.py optimizers.py train.py predict.py
    visualize.py test_integration.py requirements.txt Makefile
)
faltan=0
for archivo in "${archivos[@]}"; do
    if [[ ! -f "$RAIZ/$archivo" ]]; then
        printf '    falta: %s\n' "$archivo"
        faltan=1
    fi
done
if [[ "$faltan" -eq 0 ]]; then
    pasar "Todos los archivos necesarios estan presentes"
else
    fallar "Faltan archivos necesarios"
fi

obligatoria
explicar "Compila todos los Python sin ejecutarlos." \
    "Detecta errores de sintaxis antes de iniciar pruebas de datos o entrenamiento."
referencia "network.py" "class Network"
prueba "Sintaxis Python valida"
guia_manual "python3 -m py_compile *.py" "No aparece ningun traceback y el comando termina correctamente."
if python3 -m py_compile "$RAIZ"/*.py >"$LOGS/sintaxis.log" 2>&1; then
    pasar "Todos los archivos Python compilan"
else
    fallar "La comprobacion de sintaxis fallo"
    sed -n '1,35p' "$LOGS/sintaxis.log"
fi

cabecera "2 · RESTRICCIONES Y DATASET"
obligatoria
explicar "Analiza imports reales con el modulo ast de Python, no con una busqueda textual." \
    "Un comentario que mencione PyTorch no es una dependencia; un import si lo es."
prueba "No se importan frameworks neuronales prohibidos"
guia_manual "grep -RniE 'tensorflow|keras|torch|pytorch|sklearn' *.py requirements.txt" "No aparecen imports prohibidos; las menciones documentales deben revisarse manualmente."
if python3 - "$RAIZ" <<'PY' >"$LOGS/imports.log" 2>&1
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in root.glob("*.py"):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name.split(".")[0] for item in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        forbidden = set(names) & {"tensorflow", "keras", "torch", "pytorch", "sklearn"}
        if forbidden:
            raise SystemExit(f"{path.name}:{node.lineno}: {sorted(forbidden)}")
print("No hay imports prohibidos")
PY
then
    pasar "$(cat "$LOGS/imports.log")"
else
    fallar "Se encontro un framework prohibido"
    cat "$LOGS/imports.log"
fi

obligatoria
explicar "Lee todas las filas y valida IDs, etiquetas, dimensiones y numeros." \
    "Confirma que el entrenamiento recibe el dataset esperado: 569 casos, 30 features y dos clases."
referencia "preprocessing.py" "def load_data"
prueba "Dataset valido: 569 muestras, 30 features y etiquetas B/M"
guia_manual "wc -l data.csv; awk -F, 'NF != 32 {print NR, NF}' data.csv" "569 filas y ninguna fila con un numero de columnas distinto de 32."
if python3 - "$RAIZ/data.csv" <<'PY' >"$LOGS/dataset.log" 2>&1
import csv
import math
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.reader(stream))
if len(rows) != 569:
    raise SystemExit(f"se esperaban 569 filas y hay {len(rows)}")
if any(len(row) != 32 for row in rows):
    raise SystemExit("cada fila debe contener id, etiqueta y 30 features")
if {row[1] for row in rows} != {"B", "M"}:
    raise SystemExit("las etiquetas deben ser B y M")
if len({row[0] for row in rows}) != len(rows):
    raise SystemExit("hay IDs repetidos")
for row in rows:
    for value in row[2:]:
        if not math.isfinite(float(value)):
            raise SystemExit("hay una feature no numerica o no finita")
print("569 filas; 30 features numericas; etiquetas B/M; IDs unicos")
PY
then
    pasar "$(cat "$LOGS/dataset.log")"
else
    fallar "El dataset no cumple el contrato esperado"
    cat "$LOGS/dataset.log"
fi

cabecera "3 · DIVISION Y TEST DE INTEGRACION"
copiar_fuentes
obligatoria
prueba "Division reproducible 80/20 y sin perdida de filas"
explicar "Divide con seed 42, comprueba 455/114 filas y repite en otra pareja de archivos." \
    "La misma entrada debe producir la misma particion; los dos conjuntos deben conservar todas las muestras."
referencia "split_dataset.py" "def split_dataset"
guia_manual "python3 split_dataset.py data.csv --output_train data_training.csv --output_valid data_validation.csv; wc -l data_training.csv data_validation.csv" "Se generan 455 filas de entrenamiento y 114 de validacion."
ejecutar "Division reproducible 80/20 y sin perdida de filas" bash -c "
    cd '$TMP' &&
    python3 split_dataset.py data.csv --output_train train-a.csv --output_valid valid-a.csv >/dev/null &&
    python3 split_dataset.py data.csv --output_train train-b.csv --output_valid valid-b.csv >/dev/null &&
    test \"\$(wc -l < train-a.csv)\" -eq 455 &&
    test \"\$(wc -l < valid-a.csv)\" -eq 114 &&
    cmp train-a.csv train-b.csv &&
    cmp valid-a.csv valid-b.csv
"

obligatoria
explicar "Prueba que el CLI rechaza un ratio de entrenamiento fuera de (0, 1)." \
    "Una validacion de entrada correcta evita generar conjuntos imposibles o resultados silenciosamente invalidos."
referencia "split_dataset.py" "train_ratio"
prueba "Rechazo de ratio de division invalido"
guia_manual "python3 split_dataset.py data.csv --train_ratio 1.0" "El programa devuelve un error y un codigo de salida distinto de 0."
if (cd "$TMP" && python3 split_dataset.py data.csv --train_ratio 1.0 \
    >/dev/null 2>"$LOGS/ratio.log"); then
    fallar "Se acepto un train_ratio invalido"
else
    pasar "El CLI rechaza train_ratio=1.0"
fi

obligatoria
prueba "Test de integracion existente"
explicar "Ejecuta el flujo de integracion que ya existe en el proyecto." \
    "Comprueba conjuntamente preprocesamiento, red, entrenamiento, guardado/carga y metricas."
referencia "test_integration.py" "def main"
guia_manual "python3 split_dataset.py data.csv; python3 test_integration.py" "El resumen de integracion termina sin errores."
ejecutar "Test de integracion existente" bash -c "
    cd '$TMP' &&
    python3 split_dataset.py data.csv --output_train data_training.csv \
        --output_valid data_validation.csv >/dev/null &&
    python3 test_integration.py
"

obligatoria
prueba "La validacion es obligatoria durante el entrenamiento"
explicar "Intenta entrenar sin un archivo de validacion existente y exige un error." \
    "Evita que una ejecucion incompleta se presente como un entrenamiento valido del subject."
referencia "train.py" "Cargar datos de validación"
guia_manual "python3 train.py --dataset data_training.csv --validation no-existe.csv --epochs 1" "El proceso termina con un codigo distinto de 0."
if (cd "$TMP" && python3 train.py --dataset train-a.csv \
    --validation no-existe.csv --layers 4 4 --epochs 1 \
    >/dev/null 2>"$LOGS/validacion-obligatoria.log"); then
    fallar "El entrenamiento acepto una validacion inexistente"
else
    pasar "El entrenamiento rechaza una validacion inexistente"
fi

obligatoria
prueba "Gradientes verificados numericamente"
explicar "Compara los gradientes de backpropagation con diferencias finitas en una red pequena." \
    "Una red puede entrenar y aun tener derivadas incorrectas; esta prueba valida la matematica central."
referencia "network.py" "def _backward_batch"
guia_manual "python3 test_gradient_check.py" "El error maximo absoluto de los gradientes es menor que 1e-5."
if (cd "$TMP" && python3 - <<'PY' >"$LOGS/gradient-check.log" 2>&1
from network import Network

network = Network(loss='categorical_crossentropy', learning_rate=0.01)
network.build([2, 3, 2], ['tanh', 'softmax'], weight_init='xavier', seed=7)
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
    raise SystemExit(f'error maximo demasiado alto: {maximum_error}')
print(f'error maximo absoluto: {maximum_error:.3e}')
PY
); then
    pasar "$(cat "$LOGS/gradient-check.log")"
else
    fallar "El gradient check no coincide"
    cat "$LOGS/gradient-check.log"
fi

obligatoria
prueba "Early stopping conserva el mejor estado"
explicar "Entrena con paciencia reducida y verifica que se restaura la mejor epoca antes de guardar." \
    "El modelo final debe corresponder al mejor resultado de validacion, no a la ultima epoca ejecutada."
referencia "train.py" "best_model_state"
guia_manual "python3 train.py --dataset data_training.csv --validation data_validation.csv --early_stopping --patience 1" "La salida indica que se restauran los pesos de la mejor epoca y se crea el modelo."
if (cd "$TMP" && python3 train.py --dataset train-a.csv --validation valid-a.csv \
    --layers 8 8 --epochs 20 --learning_rate 0.05 --batch_size 16 \
    --activation relu --weight_init he --early_stopping --patience 1 \
    --save early-stop.json --history early-stop-history.json --seed 42 \
    >"$LOGS/early-stopping.log" 2>&1 &&
    grep -q "Restaurados pesos de la mejor" "$LOGS/early-stopping.log" &&
    test -s early-stop.json); then
    pasar "Se restauran los pesos de la mejor epoca"
else
    fallar "No se pudo demostrar la restauracion del mejor estado"
    sed -n '1,35p' "$LOGS/early-stopping.log"
fi

cabecera "4 · ENTRENAMIENTO Y MODELO"
obligatoria
prueba "Entrenamiento corto con modelo e historial"
explicar "Entrena 20 epocas en un directorio aislado y exige modelo e historial no vacios." \
    "Prueba el camino real del CLI, no solo imports o funciones unitarias."
referencia "train.py" "def train_epoch"
guia_manual "python3 train.py --dataset data_training.csv --validation data_validation.csv --layers 16 16 --epochs 20 --save model.json --history history.json" "Se crean model.json e history.json."
ejecutar "Entrenamiento corto con modelo e historial" bash -c "
    cd '$TMP' &&
    python3 train.py --dataset train-a.csv --validation valid-a.csv \
        --layers 16 16 --epochs 20 --learning_rate 0.05 \
        --batch_size 16 --activation relu --weight_init he \
        --early_stopping --save model.json --history history.json --seed 42 >/dev/null &&
    test -s model.json && test -s history.json
"

obligatoria
explicar "Inspecciona el JSON producido por el entrenamiento." \
    "Verifica arquitectura, pesos, biases y parametros de preprocesamiento, no solo que exista un archivo."
referencia "utils.py" "def save_model"
prueba "Esquema del modelo y minimo de dos capas ocultas"
guia_manual "python3 -m json.tool model.json | less" "Se observan architecture, activations, weights, biases y preprocessing."
if python3 - "$TMP/model.json" <<'PY' >"$LOGS/modelo.log" 2>&1
import json
import sys

with open(sys.argv[1]) as stream:
    model = json.load(stream)
required = {"architecture", "activations", "weights", "biases", "preprocessing"}
missing = required.difference(model)
if missing:
    raise SystemExit(f"faltan claves: {sorted(missing)}")
architecture = model["architecture"]
if architecture[0] != 30 or architecture[-1] != 2 or len(architecture) < 4:
    raise SystemExit(f"arquitectura no valida: {architecture}")
if model["activations"][-1] != "softmax":
    raise SystemExit("la salida no usa softmax")
if len(model["weights"]) != len(architecture) - 1:
    raise SystemExit("numero de matrices de pesos incoherente")
print(f"arquitectura {architecture}; softmax; esquema completo")
PY
then
    pasar "$(cat "$LOGS/modelo.log")"
else
    fallar "El modelo guardado no cumple el contrato"
    cat "$LOGS/modelo.log"
fi

obligatoria
explicar "Evalua la Binary Cross-Entropy con un caso conocido." \
    "Comprueba numericamente la metrica exigida, en vez de aceptar cualquier numero mostrado por pantalla."
referencia "loss.py" "def binary_crossentropy"
prueba "Binary Cross-Entropy numericamente correcta"
guia_manual "python3 -c 'from loss import binary_crossentropy; print(binary_crossentropy([1,0], [0.8,0.3]))'" "Se obtiene aproximadamente 0.2899092476."
if (cd "$RAIZ" && python3 - <<'PY') >"$LOGS/bce.log" 2>&1
import math
from loss import binary_crossentropy

value = binary_crossentropy([1, 0], [0.8, 0.3])
expected = -(math.log(0.8) + math.log(0.7)) / 2
if abs(value - expected) > 1e-10:
    raise SystemExit(f"valor {value} distinto de {expected}")
print(f"BCE={value:.10f}")
PY
then
    pasar "$(cat "$LOGS/bce.log")"
else
    fallar "Binary Cross-Entropy incorrecta"
    cat "$LOGS/bce.log"
fi

cabecera "5 · PREDICCION Y METRICAS"
obligatoria
prueba "Prediccion sobre validacion y CSV de salida"
explicar "Carga el modelo entrenado, reutiliza su preprocesamiento y genera una fila por muestra." \
    "Valida la interfaz de prediccion y que las probabilidades sean utilizables."
referencia "predict.py" "def save_predictions"
guia_manual "python3 predict.py --dataset data_validation.csv --model model.json --output predictions.csv; wc -l predictions.csv" "Se crea predictions.csv con cabecera y 114 predicciones."
ejecutar "Prediccion sobre validacion y CSV de salida" bash -c "
    cd '$TMP' &&
    python3 predict.py --dataset valid-a.csv --model model.json \
        --output predictions.csv >/dev/null &&
    test \"\$(wc -l < predictions.csv)\" -eq 115 &&
    grep -q '^id,true_label,predicted_label,probability_benign,probability_malignant$' predictions.csv
"

obligatoria
explicar "Comprueba etiquetas, probabilidades entre 0 y 1 y suma de probabilidades igual a 1." \
    "Un CSV puede existir y seguir conteniendo predicciones corruptas; esta prueba valida su contenido."
prueba "Predicciones con formato y probabilidades validas"
guia_manual "awk -F, 'NR > 1 {print \$4+\$5}' predictions.csv | sort -u" "Las probabilidades suman aproximadamente 1 y estan entre 0 y 1."
if python3 - "$TMP/predictions.csv" <<'PY' >"$LOGS/predicciones.log" 2>&1
import csv
import math
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 114:
    raise SystemExit("numero de predicciones incorrecto")
for row in rows:
    if row["true_label"] not in {"B", "M"} or row["predicted_label"] not in {"B", "M"}:
        raise SystemExit("etiqueta invalida")
    benign = float(row["probability_benign"])
    malignant = float(row["probability_malignant"])
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (benign, malignant)):
        raise SystemExit("probabilidad fuera de [0, 1]")
    if abs(benign + malignant - 1) > 1e-5:
        raise SystemExit("las probabilidades no suman 1")
print("114 predicciones validas; probabilidades finitas y normalizadas")
PY
then
    pasar "$(cat "$LOGS/predicciones.log")"
else
    fallar "El contenido de las predicciones no es valido"
    cat "$LOGS/predicciones.log"
fi

cabecera "6 · VISUALIZACION"
obligatoria
prueba "Curvas loss y accuracy en backend no interactivo"
explicar "Genera las curvas sin abrir una ventana grafica." \
    "Permite validar visualizacion en CI, SSH y servidores sin escritorio."
referencia "visualize.py" "def plot_loss_curve"
guia_manual "MPLBACKEND=Agg python3 visualize.py --history training_history.json --loss_output loss_curve.png --accuracy_output accuracy_curve.png; ls -lh *.png" "Se crean loss_curve.png y accuracy_curve.png con tamano mayor que cero."
ejecutar "Curvas loss y accuracy en backend no interactivo" bash -c "
    cd '$TMP' &&
    MPLBACKEND=Agg python3 visualize.py --history history.json \
        --loss_output loss.png --accuracy_output accuracy.png >/dev/null &&
    test -s loss.png && test -s accuracy.png
"
if [[ -s "$TMP/loss.png" ]]; then
    abrir_grafico "$TMP/loss.png"
    printf '  %bQue estas viendo:%b eje X=epocas; eje Y=perdida. Las curvas training y validation deberian tender a bajar. Una validation que empeora mientras training mejora puede indicar overfitting.\n' "$CIRAN" "$RESET"
fi
if [[ -s "$TMP/accuracy.png" ]]; then
    abrir_grafico "$TMP/accuracy.png"
    printf '  %bQue estas viendo:%b eje X=epocas; eje Y=accuracy. Las curvas training y validation deberian subir y mantenerse razonablemente proximas; una gran separacion sugiere overfitting.\n' "$CIRAN" "$RESET"
fi

cabecera "7 · BONUS EJECUTABLES"
explicar "Ejecuta configuraciones bonus reales cuando el CLI las ofrece." \
    "Buscar palabras en el codigo solo demuestra que se mencionan; ejecutar demuestra que funcionan."
prueba "Optimizador Adam"
guia_manual "python3 train.py --dataset data_training.csv --optimizer adam --epochs 2 --save adam.json" "El entrenamiento termina y crea adam.json."
if (cd "$TMP" && python3 train.py --dataset train-a.csv --validation valid-a.csv \
    --layers 8 8 --epochs 2 --learning_rate 0.005 --batch_size 32 \
    --activation relu --weight_init he --optimizer adam --save adam.json \
    --history adam-history.json --seed 42 >/dev/null 2>&1); then
    BONUS_PASADOS=$((BONUS_PASADOS + 1))
    pasar "Adam ejecutado correctamente"
else
    avisar "Adam no se pudo ejecutar; el bonus no afecta a las obligatorias"
fi

explicar "Comprueba que Dropout aplica la mascara y el escalado tambien al gradiente." \
    "Un smoke test solo demuestra que entrena; esta prueba verifica el comportamiento matematico del bonus."
prueba "Dropout"
referencia "network.py" "dropout_mask"
guia_manual "python3 train.py --dataset data_training.csv --dropout 0.2 --epochs 2 --save dropout.json" "El entrenamiento termina y crea dropout.json."
if (cd "$TMP" && python3 train.py --dataset train-a.csv --validation valid-a.csv \
    --layers 8 8 --epochs 2 --learning_rate 0.01 --batch_size 32 \
    --activation relu --weight_init he --dropout 0.2 --save dropout.json \
    --history dropout-history.json --seed 42 >/dev/null 2>&1); then
    BONUS_PASADOS=$((BONUS_PASADOS + 1))
    pasar "Dropout ejecutado correctamente"
else
    avisar "Dropout no se pudo ejecutar; el bonus no afecta a las obligatorias"
fi

explicar "Comprueba con una red pequena que las neuronas descartadas no reciben gradiente." \
    "La mascara de Dropout debe anular y escalar el gradiente de las capas ocultas durante backpropagation."
prueba "Gradiente de Dropout matematicamente correcto"
referencia "network.py" "current_layer.dropout_mask"
guia_manual "python3 test_dropout_gradient.py" "Las filas de gradiente correspondientes a mascaras cero son cero."
if (cd "$TMP" && python3 - <<'PY' >"$LOGS/dropout-gradient.log" 2>&1
from network import Network

network = Network(loss='categorical_crossentropy', learning_rate=0.01)
network.build([2, 3, 2], ['relu', 'softmax'], weight_init='he',
              dropout_rate=0.5, seed=42)
network.train_step([[1.0, 1.0]], [1])
hidden = network.layers[0]
mask = hidden.dropout_mask[0]
for neuron_index, active in enumerate(mask):
    if active == 0.0 and any(abs(value) > 1e-12 for value in hidden.dW[neuron_index]):
        raise SystemExit("una neurona descartada conserva gradiente")
if not any(active == 0.0 for active in mask):
    raise SystemExit("la semilla no produjo ninguna neurona descartada")
print("mascara y gradiente de Dropout verificados")
PY
); then
    BONUS_PASADOS=$((BONUS_PASADOS + 1))
    pasar "$(cat "$LOGS/dropout-gradient.log")"
else
    avisar "El gradiente de Dropout no supera la comprobacion matematica"
    cat "$LOGS/dropout-gradient.log"
fi

cabecera "INFORME FINAL"
printf '  %bPruebas totales:%b %d\n' "$CIRAN" "$RESET" "$TOTAL"
printf '  %bObligatorias:%b   %d\n' "$CIRAN" "$RESET" "$OBLIGATORIAS"
printf '  %bPasadas:%b       %d\n' "$VERDE" "$RESET" "$PASADAS"
printf '  %bFallidas:%b      %d\n' "$ROJO" "$RESET" "$FALLIDAS"
printf '  %bAvisos:%b        %d\n' "$AMARILLO" "$RESET" "$AVISOS"
printf '  %bBonus ejecutados:%b %d\n' "$CIRAN" "$RESET" "$BONUS_PASADOS"

if [[ "$FALLIDAS" -eq 0 ]]; then
    printf '\n%b✅ Todas las comprobaciones obligatorias han pasado.%b\n' "$VERDE" "$RESET"
    printf '%bNota:%b pasar esta suite no sustituye una defensa ni demuestra ausencia absoluta de bugs.\n' "$AMARILLO" "$RESET"
    finalizar
    exit 0
fi

printf '\n%b❌ Hay comprobaciones fallidas. Revisa los diagnosticos anteriores.%b\n' "$ROJO" "$RESET"
finalizar
exit 1
