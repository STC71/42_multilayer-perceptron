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
DETALLE_EJEMPLO=""
DETALLE_FORMULA=""
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

concepto() {
    local titulo="$1"
    local cotidiano="$2"
    local matematicas="$3"
    local codigo="$4"
    local defensa="$5"
    local archivo="${6:-}"
    local patron="${7:-}"
    printf '\n  %b%s%b\n' "$NEGRITA" "$titulo" "$RESET"
    printf '  %bEjemplo cotidiano:%b %s\n' "$CIRAN" "$RESET" "$cotidiano"
    printf '  %bMatematicas:%b %s\n' "$CIRAN" "$RESET" "$matematicas"
    if [[ -n "$archivo" && -n "$patron" ]]; then
        referencia "$archivo" "$patron"
        printf '  %bEn este proyecto:%b %s (%s)\n' "$CIRAN" "$RESET" "$codigo" "$DETALLE_IMPLEMENTACION"
    else
        printf '  %bEn este proyecto:%b %s\n' "$CIRAN" "$RESET" "$codigo"
    fi
    printf '  %bPregunta de defensa:%b %s\n' "$CIRAN" "$RESET" "$defensa"
}

explicar() {
    DETALLE_QUE="$1"
    DETALLE_POR_QUE="$2"
    DETALLE_EJEMPLO="${3:-}"
    DETALLE_FORMULA="${4:-}"
    DETALLE_IMPLEMENTACION=""
}

guia_manual() {
    printf '  %bQue hace:%b %s\n' "$CIRAN" "$RESET" "$DETALLE_QUE"
    printf '  %bPor que:%b %s\n' "$CIRAN" "$RESET" "$DETALLE_POR_QUE"
    if [[ -n "$DETALLE_EJEMPLO" ]]; then
        printf '  %bEjemplo sencillo:%b %s\n' "$CIRAN" "$RESET" "$DETALLE_EJEMPLO"
    fi
    if [[ -n "$DETALLE_FORMULA" ]]; then
        printf '  %bFormula clave:%b %s\n' "$CIRAN" "$RESET" "$DETALLE_FORMULA"
    fi
    printf '  %bImplementacion:%b %s\n' "$CIRAN" "$RESET" "${DETALLE_IMPLEMENTACION:-No se ha definido un simbolo unico: la prueba se basa en el contrato observable y se valida por su resultado real.}"
    printf '  %bRepetir manualmente:%b %b%s%b\n' "$CIRAN" "$RESET" "$VERDE" "$1" "$RESET"
    printf '  %bResultado esperado:%b %s\n' "$CIRAN" "$RESET" "$2"
}

referencia() {
    local archivo="$1"
    local patron="$2"
    local linea
    if [[ ! -f "$RAIZ/$archivo" ]]; then
        DETALLE_IMPLEMENTACION="Archivo no localizado: $archivo"
        return
    fi
    linea="$(grep -nE -- "$patron" "$RAIZ/$archivo" 2>/dev/null | head -1 | cut -d: -f1)"
    if [[ -n "$linea" ]]; then
        DETALLE_IMPLEMENTACION="$archivo:$linea"
    else
        DETALLE_IMPLEMENTACION="No localizada: el patron '$patron' no aparece en $archivo. La prueba sigue validando el comportamiento observable; si el codigo cambia, disminuye la trazabilidad y conviene revisar este patron."
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
    visualize.py test_integration.py test_gradient_check.py
    test_dropout_gradient.py requirements.txt Makefile
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
    "La misma entrada debe producir la misma particion; los dos conjuntos deben conservar todas las muestras." \
    "Como repartir 10 fichas: 8 van a practicar y 2 quedan para comprobar si aprendimos, sin repetir una ficha." \
    "n_train = floor(569 x 0.8) = 455; n_validation = 569 - 455 = 114"
referencia "split_dataset.py" "def split_dataset"
guia_manual "python3 split_dataset.py data.csv --output_train data_training.csv --output_valid data_validation.csv; wc -l data_training.csv data_validation.csv" "Se generan 455 filas de entrenamiento y 114 de validacion."
ejecutar "Division reproducible 80/20 y sin perdida de filas" bash -c "
    cd '$TMP' &&
    python3 split_dataset.py data.csv --output_train train-a.csv --output_valid valid-a.csv >/dev/null &&
    python3 split_dataset.py data.csv --output_train train-b.csv --output_valid valid-b.csv >/dev/null &&
    test \"\$(wc -l < train-a.csv)\" -eq 455 &&
    test \"\$(wc -l < valid-a.csv)\" -eq 114 &&
    cmp train-a.csv train-b.csv &&
    cmp valid-a.csv valid-b.csv &&
    python3 -c 'import csv,sys; read=lambda path: {row[0] for row in csv.reader(open(path, newline=\"\"))}; train=read(sys.argv[1]); valid=read(sys.argv[2]); assert not train & valid, \"train y validation comparten IDs\"; assert len(train | valid) == 569, \"la union no conserva las 569 muestras\"; print(\"particiones disjuntas y union completa\")' train-a.csv valid-a.csv
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
    "Una red puede entrenar y aun tener derivadas incorrectas; esta prueba valida la matematica central." \
    "Como comprobar una pendiente: mover un peso un poquito arriba y abajo y observar cuanto cambia el error." \
    "dL/dtheta ≈ [L(theta+epsilon)-L(theta-epsilon)]/(2 epsilon)"
referencia "network.py" "def _backward_batch"
guia_manual "python3 test_gradient_check.py" "El error maximo absoluto de los gradientes es menor que 1e-5."
if (cd "$TMP" && python3 test_gradient_check.py >"$LOGS/gradient-check.log" 2>&1); then
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
    "Verifica arquitectura, pesos, biases y parametros de preprocesamiento, no solo que exista un archivo." \
    "Guardar un modelo es como guardar tanto el mapa como las reglas y herramientas necesarias para repetir un viaje." \
    "z = W x + b; la arquitectura debe concordar con las dimensiones de W y b"
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
explicar "Distingue la activacion oculta por defecto de la activacion elegida explicitamente por esta suite." \
    "Evita responder de forma ambigua en la defensa: la configuracion por defecto usa sigmoid, el entrenamiento probado aqui solicita relu y la salida conserva softmax." \
    "Como elegir el modo de conducir: el coche tiene un modo por defecto, pero el trayecto puede seleccionar otro; la marcha final sigue siendo la misma." \
    "Ocultas por defecto = sigmoid; ejecucion de esta suite = relu; salida = softmax"
referencia "config.py" "DEFAULT_HIDDEN_ACTIVATION"
prueba "Activaciones trazables: default sigmoid, suite relu y salida softmax"
guia_manual "grep -nE 'DEFAULT_HIDDEN_ACTIVATION|DEFAULT_OUTPUT_ACTIVATION' config.py; python3 train.py --help" \
    "config.py muestra sigmoid/softmax y el comando de entrenamiento de la suite usa --activation relu."
if python3 - "$TMP/model.json" "$RAIZ" <<'PY' >"$LOGS/activaciones.log" 2>&1
import json
import sys

model_path, root = sys.argv[1:3]
sys.path.insert(0, root)
import config
import train

if config.DEFAULT_HIDDEN_ACTIVATION != "sigmoid":
    raise SystemExit("la activacion oculta por defecto ya no es sigmoid")
if config.DEFAULT_OUTPUT_ACTIVATION != "softmax":
    raise SystemExit("la activacion de salida por defecto ya no es softmax")
sys.argv = ["train.py", "--dataset", "train-a.csv", "--validation", "valid-a.csv"]
args = train.parse_arguments()
if args.activation != config.DEFAULT_HIDDEN_ACTIVATION:
    raise SystemExit(f"el parser usa {args.activation}, no el default {config.DEFAULT_HIDDEN_ACTIVATION}")
with open(model_path) as stream:
    model = json.load(stream)
hidden = model["activations"][:-1]
if set(hidden) != {"relu"} or model["activations"][-1] != "softmax":
    raise SystemExit(f"modelo de la suite inesperado: {model['activations']}")
print("default oculto=sigmoid; suite oculto=relu; salida=softmax")
PY
then
    pasar "$(cat "$LOGS/activaciones.log")"
else
    fallar "La trazabilidad de activaciones no coincide"
    cat "$LOGS/activaciones.log"
fi

obligatoria
explicar "Evalua la Binary Cross-Entropy con un caso conocido." \
    "Comprueba numericamente la metrica exigida, en vez de aceptar cualquier numero mostrado por pantalla." \
    "Decir 90% con mucha seguridad cuando la respuesta era falsa debe doler mas que dudar con un 55%." \
    "BCE = -1/N sum[y log(p) + (1-y) log(1-p)]"
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
    "Valida la interfaz de prediccion y que las probabilidades sean utilizables." \
    "Es como usar la misma regla con la que estudiamos: si cambiamos la escala al medir, la prediccion deja de ser comparable." \
    "softmax(z_i) = exp(z_i) / sum_j exp(z_j); sum_i p_i = 1"
referencia "predict.py" "def save_predictions"
guia_manual "python3 predict.py --dataset data_validation.csv --model model.json --output predictions.csv; wc -l predictions.csv" "Se crea predictions.csv con cabecera y 114 predicciones."
ejecutar "Prediccion sobre validacion y CSV de salida" bash -c "
    cd '$TMP' &&
    python3 predict.py --dataset valid-a.csv --model model.json \
        --output predictions.csv >predict.log &&
    test \"\$(wc -l < predictions.csv)\" -eq 115 &&
    grep -q '^id,true_label,predicted_label,probability_benign,probability_malignant$' predictions.csv &&
    grep -Eq 'Binary Cross-Entropy: [0-9]+(\\.[0-9]+)?' predict.log
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

obligatoria
explicar "Calcula las metricas con una matriz de confusion conocida y compara cada resultado con su formula." \
    "Accuracy resume aciertos, precision evita celebrar falsos positivos, recall evita perder positivos reales y F1 equilibra precision y recall." \
    "Con TP=2, TN=2, FP=1 y FN=1: accuracy=4/6, precision=2/3, recall=2/3 y F1=2/3." \
    "Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2PR/(P+R)" \
    >/dev/null
referencia "metrics.py" "def compute_accuracy"
prueba "Accuracy, precision, recall y F1 numericamente correctas"
guia_manual "python3 -c 'from metrics import compute_all_metrics; print(compute_all_metrics([1,1,1,0,0,0], [[.1,.9],[.8,.2],[.2,.8],[.4,.6],[.8,.2],[.7,.3]]))'" \
    "Se obtiene accuracy=0.666..., precision=0.666..., recall=0.666... y F1=0.666...."
if python3 - "$RAIZ" <<'PY' >"$LOGS/metricas.log" 2>&1
import math
import sys

sys.path.insert(0, sys.argv[1])
from metrics import compute_all_metrics

y_true = [1, 1, 1, 0, 0, 0]
y_pred = [[0.1, 0.9], [0.8, 0.2], [0.2, 0.8],
          [0.4, 0.6], [0.8, 0.2], [0.7, 0.3]]
result = compute_all_metrics(y_true, y_pred)
expected = 2 / 3
for key in ("accuracy", "precision", "recall", "f1_score"):
    if not math.isclose(result[key], expected, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(f"{key}={result[key]} distinto de {expected}")
if result["confusion_matrix"] != [[2, 1], [1, 2]]:
    raise SystemExit(f"matriz inesperada: {result['confusion_matrix']}")
print("matriz [[TN=2, FP=1], [FN=1, TP=2]]; cuatro metricas correctas")
PY
then
    pasar "$(cat "$LOGS/metricas.log")"
else
    fallar "Las metricas no coinciden con el caso conocido"
    cat "$LOGS/metricas.log"
fi

obligatoria
explicar "Comprueba que el historial tiene las cuatro series exigidas y que cada epoca contiene valores finitos." \
    "Un grafico puede existir aunque el entrenamiento no haya registrado correctamente loss y accuracy de train y validation."
prueba "Historial completo y numericamente valido"
guia_manual "python3 -c 'import json; print(json.load(open(\"history.json\")).keys())'" \
    "El historial contiene loss, val_loss, accuracy y val_accuracy con la misma longitud."
if python3 - "$TMP/history.json" <<'PY' >"$LOGS/historial.log" 2>&1
import json
import math
import sys

with open(sys.argv[1]) as stream:
    history = json.load(stream)
required = ("loss", "val_loss", "accuracy", "val_accuracy")
if any(key not in history for key in required):
    raise SystemExit(f"faltan series: {required}")
lengths = {len(history[key]) for key in required}
if len(lengths) != 1 or not lengths or not next(iter(lengths)):
    raise SystemExit("las series no tienen una longitud comun positiva")
if any(not math.isfinite(float(value)) for key in required for value in history[key]):
    raise SystemExit("el historial contiene valores no finitos")
if any(not 0 <= float(value) <= 1 for key in ("accuracy", "val_accuracy") for value in history[key]):
    raise SystemExit("accuracy fuera de [0, 1]")
print(f"4 series completas; {next(iter(lengths))} epocas; valores finitos")
PY
then
    pasar "$(cat "$LOGS/historial.log")"
else
    fallar "El historial no cumple el contrato"
    cat "$LOGS/historial.log"
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
    printf '  %bQue estas viendo:%b eje X=epocas; eje Y=loss (error). Azul=training y rojo=validation. Ambas deberian tender a bajar; si training baja pero validation sube, la red puede estar memorizando train (overfitting). Una loss cercana a cero significa predicciones probabilisticas muy seguras y correctas, no necesariamente buena generalizacion.\n' "$CIRAN" "$RESET"
fi
if [[ -s "$TMP/accuracy.png" ]]; then
    abrir_grafico "$TMP/accuracy.png"
    printf '  %bQue estas viendo:%b eje X=epocas; eje Y=accuracy. Azul=training, rojo=validation y linea verde discontinua=100%%. Ambas deberian subir; una gran separacion sugiere overfitting. Accuracy responde: de todos los casos, ¿cuantos clasificamos bien?\n' "$CIRAN" "$RESET"
fi

cabecera "7 · GUIA PROFUNDA PARA LA DEFENSA"
printf '  %bEsta sección convierte cada concepto en una analogía, una ecuación, una referencia del código y una pregunta típica de defensa.%b\n' "$CIRAN" "$RESET"
concepto "Feedforward" \
    "Como pasar una solicitud por varios mostradores: cada capa recibe el resultado anterior, lo transforma y lo entrega a la siguiente." \
    "z^(l)=W^(l)a^(l-1)+b^(l); a^(l)=f(z^(l)). No se modifican pesos durante este paso." \
    "La entrada normalizada atraviesa las capas hasta producir [p(B), p(M)]." \
    "¿Por qué no se actualizan los pesos durante feedforward?" \
    "network.py" "def forward"
concepto "Derivadas" \
    "Como medir cuánto cambia el volumen de una alarma si subimos un poco una señal: indica sensibilidad local." \
    "La derivada de ReLU es 1 si z>0 y 0 si z<=0; la regla de la cadena multiplica sensibilidades capa a capa." \
    "activation.py define activaciones y derivadas; el gradient check las contrasta con diferencias finitas." \
    "¿Qué ocurre con el gradiente de una neurona ReLU apagada?" \
    "activation.py" "def relu_derivative"
concepto "Backpropagation" \
    "Si una entrega llegó al lugar equivocado, recorremos el camino al revés para saber qué mostrador contribuyó al error y cuánto." \
    "Calcula dL/dW y dL/db desde la salida hacia la entrada usando la regla de la cadena." \
    "network.py conserva activaciones y calcula gradientes; el gradient check incluye pesos y biases." \
    "¿Por qué hay que guardar las activaciones del forward?" \
    "network.py" "def backward"
concepto "Gradient descent" \
    "Como bajar una montaña con pasos pequeños: miramos la pendiente y caminamos en sentido contrario al mínimo." \
    "theta_nuevo = theta - alpha * gradiente. alpha es learning rate; un valor excesivo puede saltar el mínimo." \
    "Cada batch llama a train_step y el optimizador actualiza pesos y biases." \
    "¿Qué diferencia hay entre el gradiente y el learning rate?" \
    "network.py" "def update_weights"
concepto "Normalización" \
    "Como convertir metros, euros y gramos a una escala comparable antes de combinarlos." \
    "Estandarización: x'=(x-media)/desviacion. La media y desviación se ajustan solo con train." \
    "preprocessing.py ajusta el preprocesador en train y reutiliza sus parámetros en validation y prediction." \
    "¿Por qué no se debe ajustar la media de validation con sus propios datos?" \
    "preprocessing.py" "def fit_transform"
concepto "Softmax" \
    "Como transformar puntuaciones de una votación en porcentajes que suman exactamente 100%." \
    "softmax(z_i)=exp(z_i)/sum_j exp(z_j). La salida queda en [0,1] y suma 1." \
    "La última capa usa softmax y los tests del modelo y de predicciones verifican su normalización." \
    "¿Por qué softmax es apropiado para elegir entre B y M?" \
    "activation.py" "def softmax"
concepto "Binary Cross-Entropy (BCE)" \
    "Penaliza mucho una respuesta muy segura cuando es equivocada y poco una respuesta correcta y segura." \
    "BCE=-1/N sum[y log(p)+(1-y)log(1-p)]. Para y=1,p=0.8 y y=0,p=0.3, el valor es 0.2899092476." \
    "loss.py implementa la fórmula; la suite la verifica con un caso conocido y también en predict.py." \
    "¿Por qué usamos logaritmos y por qué se recortan probabilidades extremas?" \
    "loss.py" "def binary_crossentropy"
concepto "Train y validation" \
    "Como estudiar con ejercicios de práctica y reservar un examen distinto: no debemos memorizar las preguntas del examen." \
    "Train ajusta theta; validation estima generalización con datos no usados para actualizar pesos." \
    "split_dataset.py separa 80/20 y la suite verifica que las particiones son reproducibles, disjuntas y completas." \
    "¿Qué síntoma en las curvas sugiere overfitting?" \
    "split_dataset.py" "def split_dataset"
concepto "Bias" \
    "Como una persona optimista o pesimista: incluso con señales neutras ya tiene una pequeña predisposición." \
    "z = sum(w_j x_j) + b. El bias desplaza el punto de decisión y se aprende igual que los pesos." \
    "Cada capa guarda biases y el gradient check comprueba también dL/db." \
    "¿Qué perdería la neurona si elimináramos el bias?" \
    "layers.py" "self.bias"
concepto "Accuracy, precision, recall y F1" \
    "En un detector médico, accuracy cuenta aciertos totales; precision pregunta cuántos positivos anunciados eran realmente positivos; recall pregunta cuántos positivos reales detectamos." \
    "Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2 precision recall/(precision+recall)." \
    "metrics.py construye [[TN, FP], [FN, TP]] y calcula las métricas desde esa matriz." \
    "¿Por qué accuracy puede engañar si casi todos los casos son benignos?" \
    "metrics.py" "def compute_accuracy"
concepto "Softmax y Cross-Entropy durante el entrenamiento" \
    "Softmax reparte 100 puntos de confianza entre las clases; Cross-Entropy actúa como un profesor que penaliza repartirlos mal." \
    "Con softmax y cross-entropy, el delta de salida se simplifica a delta = prediccion - realidad." \
    "La red entrena con categorical_crossentropy para etiquetas one-hot; predict.py calcula además la BCE binaria exigida por el subject." \
    "¿Por qué no debemos confundir la loss de entrenamiento con la BCE de evaluación?" \
    "network.py" "compute_loss"
concepto "Gradient descent frente a Adam" \
    "SGD es un senderista que sigue la pendiente con pasos parecidos; Adam recuerda la dirección y adapta la zancada al terreno." \
    "Adam combina una media móvil del gradiente y otra de su cuadrado para adaptar cada actualización." \
    "optimizers.py ofrece SGD, momentum, RMSprop y Adam; Adam se prueba como bonus ejecutable." \
    "¿Qué aporta Adam y por qué sigue siendo importante entender SGD?" \
    "optimizers.py" "class Adam"
concepto "Arquitectura e hiperparámetros" \
    "Elegir una red es como elegir una herramienta: una demasiado pequeña no resuelve el trabajo y una enorme puede memorizarlo." \
    "Capacidad, learning rate, batch size, inicialización y número de épocas cambian la optimización y la generalización." \
    "La arquitectura por defecto tiene entrada, dos capas ocultas y salida; el CLI permite variar capas, activación, batch size y learning rate." \
    "¿Cómo justificarías dos capas ocultas y un tamaño moderado para 455 muestras de train?" \
    "config.py" "DEFAULT_LAYERS"
concepto "Activación elegida en cada ejecución" \
    "Como una cámara que tiene un modo automático, pero permite escoger otro antes de hacer la foto: hay que decir qué modo se usó realmente." \
    "Las capas ocultas usan la activación del argumento --activation; la última capa se construye con softmax." \
    "config.py fija sigmoid como default; los comandos de entrenamiento de esta suite pasan --activation relu; el modelo guardado termina en softmax." \
    "¿Qué activación utilizaste: sigmoid o relu? Responde según el comando ejecutado, no según una descripción genérica." \
    "config.py" "DEFAULT_HIDDEN_ACTIVATION"
concepto "Overfitting y regularización" \
    "Es memorizar las preguntas de práctica y fallar el examen nuevo: el resultado de train mejora, pero validation empeora." \
    "Early stopping conserva el mejor estado de validation; dropout elimina conexiones aleatoriamente durante train." \
    "La suite comprueba restauración de early stopping y ejecuta Dropout como bonus; las curvas permiten observar la separación." \
    "¿Qué señal en loss y accuracy te haría detener el entrenamiento?" \
    "train.py" "best_model_state"

cabecera "8 · BONUS EJECUTABLES"
printf '  %bCobertura:%b Se ejecutan los bonus que tienen una ruta CLI o una propiedad matematica comprobable en esta suite: grafico combinado, Adam, Dropout y la propagacion de su mascara. Otros bonus documentales (activaciones, inicializadores, mini-batches y metricas) quedan cubiertos por el entrenamiento, el esquema del modelo y el test de integracion; la defensa humana debe explicar sus decisiones.\n' "$CIRAN" "$RESET"
explicar "Genera el grafico combinado bonus con loss a la izquierda y accuracy a la derecha." \
    "Permite comparar en una sola imagen si el modelo aprende y si generaliza, sin confundir las escalas distintas de loss y accuracy." \
    "Es como mirar dos relojes del mismo viaje: uno mide el error que baja y otro los aciertos que suben." \
    "Panel izquierdo: loss azul/rojo; panel derecho: accuracy azul/rojo y referencia verde al 100%"
referencia "visualize.py" "def plot_combined_curves"
prueba "Grafico combinado de aprendizaje (bonus)"
guia_manual "MPLBACKEND=Agg python3 visualize.py --history history.json --combined; ls -lh learning_curves.png" \
    "Se crea learning_curves.png con loss y accuracy, cada una con training y validation."
if (cd "$TMP" && MPLBACKEND=Agg python3 visualize.py --history history.json --combined \
    >/dev/null 2>"$LOGS/combined.log" && test -s learning_curves.png); then
    BONUS_PASADOS=$((BONUS_PASADOS + 1))
    pasar "Grafico combinado generado"
    abrir_grafico "$TMP/learning_curves.png"
    printf '  %bQue estas viendo:%b panel izquierdo=loss (menor es mejor); panel derecho=accuracy (mayor es mejor). En ambos, azul=training y rojo=validation. Si train mejora y validation se estanca o empeora, hay señal de overfitting.\n' "$CIRAN" "$RESET"
else
    avisar "No se pudo generar el grafico combinado; el bonus no afecta a las obligatorias"
    sed -n '1,35p' "$LOGS/combined.log"
fi

explicar "Ejecuta Adam con datos reales, validacion y guardado de modelo e historial." \
    "Un nombre en el codigo no prueba el bonus: esta ejecucion verifica que el CLI acepta el optimizador y completa un entrenamiento reproducible. Si falla, el bonus de Adam no debe darse por demostrado, aunque las pruebas mandatory sigan siendo validas."
prueba "Optimizador Adam"
referencia "optimizers.py" "class Adam"
guia_manual "python3 train.py --dataset data_training.csv --validation data_validation.csv --layers 8 8 --optimizer adam --epochs 2 --save adam.json --history adam-history.json" "El entrenamiento termina sin error y crea adam.json y adam-history.json."
if (cd "$TMP" && python3 train.py --dataset train-a.csv --validation valid-a.csv \
    --layers 8 8 --epochs 2 --learning_rate 0.005 --batch_size 32 \
    --activation relu --weight_init he --optimizer adam --save adam.json \
    --history adam-history.json --seed 42 >/dev/null 2>&1); then
    BONUS_PASADOS=$((BONUS_PASADOS + 1))
    pasar "Adam ejecutado correctamente"
else
    avisar "Adam no se pudo ejecutar; el bonus no afecta a las obligatorias"
fi

explicar "Ejecuta Dropout con datos reales y despues comprueba su mascara durante backpropagation." \
    "El smoke test demuestra que el CLI puede entrenar con Dropout; la prueba matematica siguiente verifica que las neuronas descartadas no reciben gradiente. Si falla, el bonus no es fiable aunque el entrenamiento termine."
prueba "Dropout"
referencia "network.py" "dropout_mask"
guia_manual "python3 train.py --dataset data_training.csv --validation data_validation.csv --layers 8 8 --dropout 0.2 --epochs 2 --save dropout.json --history dropout-history.json" "El entrenamiento termina sin error y crea dropout.json y dropout-history.json."
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
    "La mascara de Dropout debe anular y escalar el gradiente de las capas ocultas durante backpropagation. Esta comprobacion cubre la propiedad matematica que un simple smoke test no puede demostrar."
prueba "Gradiente de Dropout matematicamente correcto"
referencia "network.py" "current_layer.dropout_mask"
guia_manual "python3 test_dropout_gradient.py" "Las filas de gradiente correspondientes a mascaras cero son cero y al menos una neurona ha sido descartada."
if (cd "$TMP" && python3 test_dropout_gradient.py >"$LOGS/dropout-gradient.log" 2>&1); then
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
