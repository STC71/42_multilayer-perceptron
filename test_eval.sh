#!/usr/bin/env bash
#
# test_eva.sh - Script de evaluación asistida para el proyecto
# Multilayer Perceptron (42).
#
# Este script NO sustituye la defensa oral. Su objetivo es:
#   1. Ayudar al evaluador a comprobar de forma ordenada los puntos
#      de la escala oficial (en.eval.pdf).
#   2. Generar evidencias reproducibles.
#   3. Recordar las preguntas conceptuales que deben discutirse
#      con el código delante.
#
# Todo el texto del script está en español.
#

set -u
set -o pipefail

# =============================================================================
# CONFIGURACIÓN BÁSICA
# =============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS="${ROOT}/test_eva_artifacts"
LOG="${ARTIFACTS}/evaluation.log"

# Modo interactivo por defecto (se desactiva si no hay terminal)
INTERACTIVE=1
PAUSE=1
KEEP_ARTIFACTS=1

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Contadores
TOTAL=0
PASSED=0
FAILED=0
WARNINGS=0

# Detectar si estamos en un terminal interactivo
if [[ ! -t 0 || ! -t 1 ]]; then
    INTERACTIVE=0
    PAUSE=0
fi

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

usage() {
    cat <<'EOF'
Uso: ./test_eva.sh [opciones]

Opciones:
  --non-interactive   No hace preguntas ni pausas (modo automático)
  --no-pause          No espera a que el usuario pulse INTRO entre secciones
  --clean             Borra el directorio test_eva_artifacts al terminar
  -h, --help          Muestra esta ayuda

El script trabaja dentro de test_eva_artifacts/ y no modifica
el código fuente, data.csv ni los resultados generados en la raíz.
EOF
}

# Parseo simple de argumentos
for arg in "$@"; do
    case "$arg" in
        --non-interactive) INTERACTIVE=0; PAUSE=0 ;;
        --no-pause)        PAUSE=0 ;;
        --clean)           KEEP_ARTIFACTS=0 ;;
        -h|--help)         usage; exit 0 ;;
        *)
            printf '%bArgumento desconocido:%b %s\n' "$RED" "$RESET" "$arg" >&2
            usage
            exit 2
            ;;
    esac
done

# Crear directorio de artefactos limpio
rm -rf "$ARTIFACTS"
mkdir -p "$ARTIFACTS"
: > "$LOG"

log() {
    # Escribe tanto en pantalla (si es necesario) como en el log
    echo "$@" >> "$LOG"
}

section() {
    local title="$1"
    if [[ "$INTERACTIVE" -eq 1 && "$PAUSE" -eq 1 && "$TOTAL" -gt 0 ]]; then
        printf '\n%bPulsa INTRO para continuar...%b' "$YELLOW" "$RESET"
        read -r
    fi
    printf '\n%b================================================================%b\n' "$CYAN" "$RESET"
    printf '%b%s%b\n' "$CYAN" "$title" "$RESET"
    printf '%b================================================================%b\n\n' "$CYAN" "$RESET"
    log ""
    log "===== $title ====="
}

ok() {
    local msg="$1"
    TOTAL=$((TOTAL + 1))
    PASSED=$((PASSED + 1))
    printf '%b[OK]%b  %s\n' "$GREEN" "$RESET" "$msg"
    log "[OK] $msg"
}

fail() {
    local msg="$1"
    TOTAL=$((TOTAL + 1))
    FAILED=$((FAILED + 1))
    printf '%b[FAIL]%b %s\n' "$RED" "$RESET" "$msg"
    log "[FAIL] $msg"
}

warn() {
    local msg="$1"
    WARNINGS=$((WARNINGS + 1))
    printf '%b[AVISO]%b %s\n' "$YELLOW" "$RESET" "$msg"
    log "[AVISO] $msg"
}

info() {
    printf '%b[INFO]%b %s\n' "$BLUE" "$RESET" "$1"
    log "[INFO] $1"
}

# Busca líneas que coinciden con un patrón en uno o varios ficheros.
# Devuelve un string legible: "fichero.py:12,45 ; otro.py:3"
# Si no encuentra nada, devuelve cadena vacía.
find_lines() {
    local pattern="$1"
    shift
    local files=("$@")
    local result=""
    local f matches

    for f in "${files[@]}"; do
        [[ -f "$f" ]] || continue
        matches=$(grep -nE "$pattern" "$f" 2>/dev/null | cut -d: -f1 | head -8 | tr '\n' ',' | sed 's/,$//')
        if [[ -n "$matches" ]]; then
            if [[ -n "$result" ]]; then
                result="$result ; $(basename "$f"):$matches"
            else
                result="$(basename "$f"):$matches"
            fi
        fi
    done
    printf '%s' "$result"
}

# Variante de ok() que añade automáticamente "en fichero:lineas"
ok_at() {
    local msg="$1"
    local pattern="$2"
    shift 2
    local location
    location=$(find_lines "$pattern" "$@")
    if [[ -n "$location" ]]; then
        ok "$msg  →  $location"
    else
        ok "$msg"
    fi
}

# Limpieza de pantalla + pausa al terminar un PASO (solo modo interactivo)
end_of_step() {
    local step_name="$1"
    if [[ "$INTERACTIVE" -eq 1 ]]; then
        printf '\n%b--- Fin de %s ---%b\n' "$CYAN" "$step_name" "$RESET"
        if [[ "$PAUSE" -eq 1 ]]; then
            printf '%bPulsa INTRO para continuar al siguiente paso...%b' "$YELLOW" "$RESET"
            read -r
            clear 2>/dev/null || true
        fi
    fi
}

ask_yes_no() {
    # Pregunta interactiva. Devuelve 0 = sí, 1 = no
    local prompt="$1"
    local default="${2:-n}"   # por defecto 'n'

    if [[ "$INTERACTIVE" -eq 0 ]]; then
        # En modo no interactivo respondemos según el default
        if [[ "$default" == "y" || "$default" == "s" ]]; then
            return 0
        else
            return 1
        fi
    fi

    local answer
    while true; do
        printf '%b%s%b [s/N]: ' "$YELLOW" "$prompt" "$RESET"
        read -r answer
        answer="${answer:-$default}"
        case "${answer,,}" in
            s|si|sí|y|yes) return 0 ;;
            n|no)          return 1 ;;
            *) printf 'Por favor responde s o n.\n' ;;
        esac
    done
}

# =============================================================================
# MENSAJES INICIALES / ARRANQUE
# =============================================================================

clear 2>/dev/null || true

printf '%b' "$BOLD$CYAN"
cat <<'BANNER'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          test_eva.sh  -  Evaluación Multilayer Perceptron            ║
║                                                                      ║
║     Script de apoyo para la defensa según la escala oficial          ║
║     (en.eval.pdf). No sustituye la evaluación humana.                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
BANNER
printf '%b\n' "$RESET"

info "Directorio del proyecto : $ROOT"
info "Artefactos de evaluación: $ARTIFACTS"
info "Log detallado           : $LOG"
echo

printf '%bIMPORTANTE%b\n' "$BOLD$YELLOW" "$RESET"
cat <<'EOF'
Este script sigue la escala de evaluación oficial del proyecto.
Recuerda:

  • Debes evaluar SOLO lo que existe en el repositorio Git del estudiante.
  • Sé educado, respetuoso y constructivo en todo momento.
  • Destaca los fallos encontrados y discute cada aspecto con el evaluado.
  • Si aparece cualquier flag grave (trabajo vacío, trampa, etc.) la nota
    será 0 (o -42 en caso de trampa). Aun así se recomienda seguir
    hablando de lo que falló y cómo mejorarlo.

EOF

if [[ "$INTERACTIVE" -eq 1 ]]; then
    if ! ask_yes_no "¿Has leído el subject del proyecto antes de empezar la evaluación?" "n"; then
        printf '\n%bEs obligatorio haber leído el subject antes de evaluar.%b\n' "$RED" "$RESET"
        printf 'Por favor lee el PDF del subject y vuelve a ejecutar este script.\n\n'
        exit 1
    fi
else
    info "Modo no interactivo: se asume que el evaluador ha leído el subject."
fi

echo
info "Comenzamos la evaluación. Se generará un registro en $LOG"

# =============================================================================
# PASO 0 - Comprobaciones previas (Guidelines de la escala)
# =============================================================================

section "PASO 0 - Comprobaciones previas (Guidelines oficiales)"

cat <<'EOF'
Según la escala de evaluación debes verificar lo siguiente antes de
continuar con la implementación:

  1. Que no se hayan usado alias maliciosos para engañar al evaluador
     y hacerle evaluar un repositorio distinto al genuino.

  2. Que cualquier script facilitado para "ayudar" en la evaluación
     haya sido revisado por la otra parte.

  3. Que el evaluador haya leído el subject (ya preguntado arriba).

  4. Que conozcas los flags disponibles (Empty work, Incomplete work,
     Invalid compilation, Cheat, Crash, etc.). Si se activa un flag
     grave la nota es 0 (o -42 en caso de Cheat).

Vamos a realizar comprobaciones automáticas y algunas preguntas
interactivas.
EOF

echo

# -----------------------------------------------------------------------------
# 0.1  Comprobar que estamos dentro de un repositorio Git
# -----------------------------------------------------------------------------
info "0.1  Comprobando que el directorio actual es un repositorio Git..."

if [[ -d "$ROOT/.git" ]] || git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    ok "Se detectó un repositorio Git en $ROOT"
    REPO_URL=$(git -C "$ROOT" remote get-url origin 2>/dev/null || echo "no configurado")
    CURRENT_COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo "desconocido")
    info "    Remote origin : $REPO_URL"
    info "    Commit actual : $CURRENT_COMMIT"
else
    fail "No se encontró un repositorio Git en $ROOT"
    warn "La evaluación debe realizarse sobre el repositorio Git del estudiante."
fi

echo

# -----------------------------------------------------------------------------
# 0.2  Comprobar posibles alias peligrosos / engaños
# -----------------------------------------------------------------------------
info "0.2  Buscando posibles alias o trampas en el entorno del shell..."

# Mostrar siempre los comandos recomendados para comprobación manual
printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      type git ; type python3 ; type ls ; type make\n'
printf '      alias\n'
printf '      echo "GIT_DIR=$GIT_DIR"\n'
printf '      git remote -v ; git log -1 --oneline\n'
echo

# Lista de comandos críticos que no deberían estar aliasados de forma sospechosa
SUSPICIOUS=0

for cmd in git python python3 ls cat head tail grep make; do
    # type -t devuelve "alias" si está aliasado
    if type -t "$cmd" 2>/dev/null | grep -q alias; then
        alias_def=$(alias "$cmd" 2>/dev/null || true)
        warn "El comando '$cmd' está definido como alias: $alias_def"
        SUSPICIOUS=1
    fi
done

# Comprobar variables de entorno raras que a veces se usan para engañar
if [[ -n "${GIT_DIR:-}" && "$GIT_DIR" != "$ROOT/.git" ]]; then
    warn "La variable GIT_DIR está definida de forma inusual: $GIT_DIR"
    SUSPICIOUS=1
fi

if [[ $SUSPICIOUS -eq 0 ]]; then
    ok "No se detectaron alias sospechosos en los comandos básicos"
else
    fail "Se detectaron posibles alias o variables de entorno sospechosas"
    printf '    %bRevisa con los comandos mostrados arriba.%b\n' "$YELLOW" "$RESET"
fi

echo

# -----------------------------------------------------------------------------
# 0.3  Revisar si existen scripts de "ayuda" a la evaluación
# -----------------------------------------------------------------------------
info "0.3  Buscando scripts que puedan facilitar (o engañar) la evaluación..."

# Buscamos scripts .sh relacionados con evaluación/test (sin process substitution)
FILTERED=()
TMP_FIND="${ARTIFACTS}/found_scripts.txt"
find "$ROOT" -maxdepth 2 -type f -name '*.sh' 2>/dev/null > "$TMP_FIND" || true

while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    base=$(basename "$f")
    # Solo nos interesan los que tengan palabras clave y no sean este script ni el .bak
    if [[ "$base" == "test_eva.sh" || "$base" == "test_eval.sh.bak" ]]; then
        continue
    fi
    if echo "$base" | grep -qE '(eval|test|check|grade)'; then
        FILTERED+=("$f")
    fi
done < "$TMP_FIND"
rm -f "$TMP_FIND"

if [[ ${#FILTERED[@]} -eq 0 ]]; then
    ok "No se encontraron scripts adicionales de evaluación sospechosos"
else
    warn "Se encontraron los siguientes scripts relacionados con evaluación/test:"
    for s in "${FILTERED[@]}"; do
        printf '      - %s\n' "$s"
        log "      script encontrado: $s"
    done
    printf '\n    %bRecuerda:%b cualquier script que se ofrezca para "facilitar"\n' "$YELLOW" "$RESET"
    printf '    la evaluación debe ser revisado por ambas partes antes de usarlo.\n'
fi

echo

# -----------------------------------------------------------------------------
# 0.4  Recordatorio de flags de la escala
# -----------------------------------------------------------------------------
info "0.4  Flags disponibles en la escala de evaluación:"

cat <<'EOF'
    Los flags que puedes activar (y que ponen la nota a 0 o -42) son:

      • Empty work          → Trabajo vacío
      • Incomplete work     → Trabajo incompleto
      • Invalid compilation → No compila / no se puede ejecutar
      • Cheat               → Trampa / plagio  (-42)
      • Crash               → El programa se cae
      • Forbidden function  → Uso de funciones/librerías prohibidas

    Si activas cualquiera de estos flags (excepto Cheat) se recomienda
    seguir discutiendo con el estudiante qué falló y cómo solucionarlo.
EOF

echo
if [[ "$INTERACTIVE" -eq 1 ]]; then
    if ask_yes_no "¿Has entendido los flags y las consecuencias de activarlos?" "y"; then
        ok "El evaluador confirma conocer los flags de la escala"
    else
        warn "Repasa la sección de Guidelines y Ratings de la escala antes de continuar"
    fi
else
    ok "Modo no interactivo: se asume conocimiento de los flags de la escala"
fi

echo

# -----------------------------------------------------------------------------
# 0.5  Confirmación final del Paso 0
# -----------------------------------------------------------------------------
info "0.5  Resumen del Paso 0"

printf '\n'
if [[ $FAILED -eq 0 ]]; then
    printf '%b✓ Paso 0 superado sin fallos críticos.%b\n' "$GREEN" "$RESET"
else
    printf '%b⚠  Se detectaron %d fallo(s) en el Paso 0.%b\n' "$YELLOW" "$FAILED" "$RESET"
    printf '   Revisa los mensajes anteriores antes de continuar.\n'
fi

if [[ "$INTERACTIVE" -eq 1 ]]; then
    echo
    if ! ask_yes_no "¿Deseas continuar con la evaluación de la implementación?" "y"; then
        printf '\nEvaluación detenida por el usuario.\n'
        printf 'Log guardado en: %s\n\n' "$LOG"
        exit 0
    fi
fi

end_of_step "PASO 0 - Comprobaciones previas"

# =============================================================================
# PASO 1 - Implementation (Modularidad, Dataset split, Softmax)
# =============================================================================

section "PASO 1 - Implementation"

cat <<'EOF'
Esta sección evalúa la implementación del programa según la escala oficial:

  • Modularidad
      - ¿La red tiene al menos 2 capas ocultas por defecto?
        (profundidad mínima = 4 contando entrada + salida)
      - ¿Se puede cambiar el número de capas ocultas con argumentos o fichero?

  • Dataset split
      - ¿El dataset se divide en entrenamiento y validación?

  • Softmax
      - ¿Está implementada la función softmax en la capa de salida?

Vamos a comprobarlo de forma automática y dejando comandos manuales
para que el evaluador pueda verificarlo también a ojo.
EOF

echo

# -----------------------------------------------------------------------------
# 1.1  Modularidad – mínimo 2 capas ocultas por defecto
# -----------------------------------------------------------------------------
info "1.1  Modularidad: ¿hay al menos 2 capas ocultas por defecto?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      grep -n "DEFAULT_LAYERS" config.py\n'
printf '      grep -n -A6 "\\-\\-layers" train.py\n'
printf '      # Debe verse algo similar a: DEFAULT_LAYERS = [30, 24, 24, 2]\n'
printf '      # y --layers con default=[24, 24] (dos capas ocultas)\n'
echo

MODULARITY_OK=1

# Comprobar DEFAULT_LAYERS en config.py
if [[ -f "$ROOT/config.py" ]]; then
    if grep -qE 'DEFAULT_LAYERS\s*=\s*\[' "$ROOT/config.py"; then
        layers_line=$(grep -E 'DEFAULT_LAYERS\s*=\s*\[' "$ROOT/config.py" | head -1)
        num_elements=$(echo "$layers_line" | grep -oE '[0-9]+' | wc -l)
        if [[ "$num_elements" -ge 4 ]]; then
            ok_at "config.py define DEFAULT_LAYERS con ≥4 elementos (entrada + ≥2 ocultas + salida)" \
                  'DEFAULT_LAYERS\s*=\s*\[' "$ROOT/config.py"
            info "    Contenido: $layers_line"
        else
            fail "DEFAULT_LAYERS no parece tener suficientes capas (se encontraron $num_elements elementos)"
            MODULARITY_OK=0
        fi
    else
        fail "No se encontró DEFAULT_LAYERS en config.py"
        MODULARITY_OK=0
    fi
else
    fail "No existe config.py"
    MODULARITY_OK=0
fi

# Comprobar que train.py expone --layers con default de al menos 2 valores
if [[ -f "$ROOT/train.py" ]]; then
    if grep -qE "\-\-layers" "$ROOT/train.py"; then
        if grep -A8 -E "\-\-layers" "$ROOT/train.py" | grep -qE 'default\s*=\s*\[[0-9]+,\s*[0-9]+'; then
            ok_at "train.py permite cambiar las capas ocultas con --layers (default ≥2 valores)" \
                  '\-\-layers' "$ROOT/train.py"
        else
            if grep -A10 -E "\-\-layers" "$ROOT/train.py" | grep -q "default"; then
                ok_at "train.py expone el argumento --layers (revisar manualmente el default)" \
                      '\-\-layers' "$ROOT/train.py"
            else
                fail "train.py tiene --layers pero no se ve un default claro con ≥2 capas ocultas"
                MODULARITY_OK=0
            fi
        fi
    else
        fail "train.py no expone el argumento --layers"
        MODULARITY_OK=0
    fi
else
    fail "No existe train.py"
    MODULARITY_OK=0
fi

echo

# -----------------------------------------------------------------------------
# 1.2  Dataset split (train / validation)
# -----------------------------------------------------------------------------
info "1.2  Dataset split: ¿se divide el dataset en entrenamiento y validación?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      ls -la data_training.csv data_validation.csv 2>/dev/null || true\n'
printf '      head -5 split_dataset.py\n'
printf '      grep -n "split\|train_ratio\|validation" split_dataset.py | head -20\n'
echo

SPLIT_OK=1

if [[ -f "$ROOT/split_dataset.py" ]]; then
    ok "Existe el script split_dataset.py"
    if grep -qE "train_ratio|validation|split_index|data_training|data_validation" "$ROOT/split_dataset.py"; then
        ok_at "split_dataset.py contiene lógica de división train/validation" \
              'train_ratio|validation|split_index|data_training|data_validation' \
              "$ROOT/split_dataset.py"
    else
        fail "split_dataset.py no parece implementar la división train/validation"
        SPLIT_OK=0
    fi
else
    fail "No se encontró split_dataset.py"
    SPLIT_OK=0
fi

# Comprobar también que config.py conoce los nombres de los ficheros
if [[ -f "$ROOT/config.py" ]]; then
    if grep -qE "TRAIN_FILE|VALIDATION_FILE|data_training|data_validation" "$ROOT/config.py"; then
        ok_at "config.py referencia los ficheros de entrenamiento y validación" \
              'TRAIN_FILE|VALIDATION_FILE|data_training|data_validation' \
              "$ROOT/config.py"
    else
        warn "1.2 – config.py no parece definir TRAIN_FILE / VALIDATION_FILE (puede ser opcional)"
    fi
fi

echo

# -----------------------------------------------------------------------------
# 1.3  Softmax en la capa de salida
# -----------------------------------------------------------------------------
info "1.3  Softmax: ¿está implementada en la capa de salida?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      grep -n "def softmax" activation.py\n'
printf '      grep -n "softmax" config.py network.py train.py | head -20\n'
printf '      # Debe usarse softmax (o DEFAULT_OUTPUT_ACTIVATION = \"softmax\") en la salida\n'
echo

SOFTMAX_OK=1

if [[ -f "$ROOT/activation.py" ]]; then
    if grep -qE "def\s+softmax\s*\(" "$ROOT/activation.py"; then
        ok_at "activation.py contiene la función softmax" \
              'def\s+softmax\s*\(' "$ROOT/activation.py"
    else
        fail "No se encontró la definición de softmax en activation.py"
        SOFTMAX_OK=0
    fi
else
    fail "No existe activation.py"
    SOFTMAX_OK=0
fi

# Comprobar que se usa como activación de salida
if [[ -f "$ROOT/config.py" ]]; then
    if grep -qiE "DEFAULT_OUTPUT_ACTIVATION\s*=\s*['\"]softmax['\"]|output.*softmax|softmax.*output" "$ROOT/config.py"; then
        ok_at "config.py indica que la activación de salida por defecto es softmax" \
              "DEFAULT_OUTPUT_ACTIVATION\s*=\s*['\"]softmax['\"]|output.*softmax|softmax.*output" \
              "$ROOT/config.py"
    else
        warn "1.3 – No se ve claramente DEFAULT_OUTPUT_ACTIVATION = 'softmax' en config.py (revisar manualmente)"
    fi
fi

# Comprobar uso en network / train
if grep -qiE "softmax" "$ROOT/network.py" "$ROOT/train.py" 2>/dev/null; then
    ok_at "Se encuentra uso de 'softmax' en network.py o train.py" \
          'softmax' "$ROOT/network.py" "$ROOT/train.py"
else
    warn "1.3 – No se encontró la palabra 'softmax' en network.py / train.py (puede estar solo en activation.py)"
fi

echo

# -----------------------------------------------------------------------------
# 1.4  Resumen del Paso 1
# -----------------------------------------------------------------------------
info "1.4  Resumen del Paso 1 (Implementation)"

echo
if [[ $MODULARITY_OK -eq 1 && $SPLIT_OK -eq 1 && $SOFTMAX_OK -eq 1 ]]; then
    printf '%b✓ Paso 1 (Implementation) superado sin fallos críticos.%b\n' "$GREEN" "$RESET"
else
    printf '%b⚠  Hay puntos del Paso 1 que requieren revisión manual o corrección.%b\n' "$YELLOW" "$RESET"
fi

echo
printf '%bRecordatorio para el evaluador:%b\n' "$BOLD" "$RESET"
printf '  • Marca Yes/No en la escala según lo que hayas verificado.\n'
printf '  • Si algo no está claro, pregunta al estudiante que te muestre\n'
printf '    el código correspondiente (DEFAULT_LAYERS, --layers, split_dataset,\n'
printf '    función softmax y dónde se aplica en la capa de salida).\n'

end_of_step "PASO 1 - Implementation"

# =============================================================================
# PASO 2 - Training (Display, Metrics, Model, Saving)
# =============================================================================

section "PASO 2 - Training"

cat <<'EOF'
Esta sección evalúa la fase de entrenamiento según la escala oficial:

  • Display
      - ¿Se muestran métricas al final de cada época?
        (formato similar al del subject: epoch XX/YY - loss: ... - val_loss: ...)
      - ¿Se muestran las curvas de aprendizaje al final?
        (al menos 2 curvas: training y validation)

  • Metrics
      - ¿Hay al menos una métrica para training y otra para validation?
      - ¿Son métricas adaptadas a clasificación binaria?

  • Model
      - ¿El modelo aprende? (las métricas mejoran en general)
      - ¿El modelo no está haciendo overfitting grave?

  • Saving
      - ¿Se guarda el modelo al final del entrenamiento?
        (topología de la red + pesos)

Nota: parte de estas comprobaciones son estáticas (código). Otras
requieren ejecutar el entrenamiento y observar la salida.
EOF

echo

# -----------------------------------------------------------------------------
# 2.1  Display – métricas por época
# -----------------------------------------------------------------------------
info "2.1  Display: ¿se muestran métricas al final de cada época?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      grep -n "epoch.*loss\|val_loss\|print.*epoch" train.py | head -20\n'
printf '      # Debe aparecer algo similar a:\n'
printf '      # epoch 01/70 - loss: 0.6882 - val_loss: 0.6788 - acc: ... - val_acc: ...\n'
echo

DISPLAY_OK=1

if [[ -f "$ROOT/train.py" ]]; then
    # Buscamos un print que contenga epoch + loss (formato del subject)
    if grep -qE 'print\(.*epoch.*loss|f"epoch \{|f'\''epoch \{' "$ROOT/train.py"; then
        ok_at "train.py contiene impresión de métricas por época (epoch + loss)" \
              'print\(.*epoch.*loss|f"epoch \{|f'\''epoch \{|epoch.*/.*loss' "$ROOT/train.py"
    else
        if grep -qE 'epoch.*/.*loss|val_loss' "$ROOT/train.py"; then
            ok_at "train.py parece imprimir loss / val_loss por época (revisar formato manualmente)" \
                  'epoch.*/.*loss|val_loss' "$ROOT/train.py"
        else
            fail "No se encontró impresión clara de métricas por época en train.py"
            DISPLAY_OK=0
        fi
    fi

    # Comprobar que se manejan tanto loss de train como de validation
    if grep -qE "val_loss|val_acc|validation" "$ROOT/train.py"; then
        ok_at "train.py calcula/muestra métricas de validación" \
              'val_loss|val_acc' "$ROOT/train.py"
    else
        fail "No se ven métricas de validación (val_loss / val_acc) en train.py"
        DISPLAY_OK=0
    fi
else
    fail "No existe train.py"
    DISPLAY_OK=0
fi

echo

# -----------------------------------------------------------------------------
# 2.2  Display – curvas de aprendizaje
# -----------------------------------------------------------------------------
info "2.2  Display: ¿se generan curvas de aprendizaje (train + validation)?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      ls -la visualize.py 2>/dev/null || true\n'
printf '      grep -n "plot\|curve\|matplotlib\|loss\|accuracy" visualize.py train.py | head -25\n'
printf '      # Debe haber al menos curvas de loss y/o accuracy para train y validation\n'
echo

CURVES_OK=1

if [[ -f "$ROOT/visualize.py" ]]; then
    ok "Existe visualize.py (script de curvas)"
    if grep -qiE "plot|matplotlib|curve|subplot" "$ROOT/visualize.py"; then
        ok_at "visualize.py usa matplotlib / genera gráficos" \
              'plot|matplotlib|curve|subplot' "$ROOT/visualize.py"
    else
        warn "2.2 – visualize.py no parece usar matplotlib (revisar manualmente)"
    fi
else
    if grep -qiE "matplotlib|pyplot|plot\(" "$ROOT/train.py" 2>/dev/null; then
        ok_at "Las curvas parecen generarse desde train.py" \
              'matplotlib|pyplot|plot\(' "$ROOT/train.py"
    else
        fail "No se encontró visualize.py ni generación de curvas en train.py"
        CURVES_OK=0
    fi
fi

# Comprobar que se guardan history de train y validation
if grep -qE "history\[.val_loss.\]|history\[.val_accuracy.\]|val_loss.*append|val_acc.*append" "$ROOT/train.py" 2>/dev/null; then
    ok_at "Se guardan métricas de validation en el historial (necesario para las curvas)" \
          'history\[.val_loss.\]|history\[.val_accuracy.\]|val_loss.*append|val_acc.*append' \
          "$ROOT/train.py"
else
    warn "2.2 – No se ve claramente el guardado de val_loss/val_accuracy en history"
fi

echo

# -----------------------------------------------------------------------------
# 2.3  Metrics – métricas adaptadas a clasificación binaria
# -----------------------------------------------------------------------------
info "2.3  Metrics: ¿hay métricas de train y validation adaptadas a clasificación binaria?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      grep -n "loss\|accuracy\|cross.?entropy\|evaluate" train.py network.py loss.py | head -25\n'
printf '      # Métricas aceptables: Binary/Categorical Cross-Entropy, Accuracy, etc.\n'
echo

METRICS_OK=1

# Comprobar que existe una función de pérdida adecuada
if [[ -f "$ROOT/loss.py" ]]; then
    if grep -qiE "binary_crossentropy|categorical_crossentropy|cross.?entropy" "$ROOT/loss.py"; then
        ok_at "loss.py implementa Cross-Entropy (adecuada para clasificación)" \
              'binary_crossentropy|categorical_crossentropy|cross.?entropy' \
              "$ROOT/loss.py"
    else
        warn "2.3 – No se ve Binary/Categorical Cross-Entropy en loss.py (revisar qué pérdida se usa)"
    fi
else
    warn "2.3 – No existe loss.py (la pérdida puede estar implementada en otro fichero)"
fi

# Comprobar que se evalúa accuracy o métrica similar
if grep -qiE "accuracy|acc\b|evaluate" "$ROOT/train.py" "$ROOT/network.py" 2>/dev/null; then
    ok_at "Se calcula accuracy (o métrica de evaluación) durante el entrenamiento" \
          'accuracy|acc\b|evaluate' "$ROOT/train.py" "$ROOT/network.py"
else
    fail "No se ve cálculo de accuracy / evaluate en train.py o network.py"
    METRICS_OK=0
fi

# Debe haber métrica tanto de train como de validation
if grep -qE "train_loss|train_acc|val_loss|val_acc" "$ROOT/train.py" 2>/dev/null; then
    ok_at "Hay métricas separadas para training y validation" \
          'train_loss|train_acc|val_loss|val_acc' "$ROOT/train.py"
else
    fail "No se diferencian claramente métricas de training y de validation"
    METRICS_OK=0
fi

echo

# -----------------------------------------------------------------------------
# 2.4  Model – el modelo aprende y no overfittea de forma evidente
# -----------------------------------------------------------------------------
info "2.4  Model: ¿el modelo aprende y evita overfitting grave?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      # Ejecutar un entrenamiento corto y observar:\n'
printf '      python3 train.py --dataset data_training.csv --epochs 20 --verbose\n'
printf '      # - loss y val_loss deberían bajar en general\n'
printf '      # - si val_loss sube mucho mientras train_loss baja → posible overfitting\n'
printf '      grep -n "early_stopping\|patience\|best_val" train.py | head -15\n'
echo

MODEL_OK=1

# Comprobar existencia de early stopping (buena señal contra overfitting)
if grep -qiE "early_stopping|patience" "$ROOT/train.py" 2>/dev/null; then
    ok_at "train.py implementa early stopping (ayuda a evitar overfitting)" \
          'early_stopping|patience' "$ROOT/train.py"
else
    warn "2.4 – No se ve early stopping (no es obligatorio, pero ayuda contra overfitting)"
fi

# Comprobar que se guarda el mejor modelo según validation
if grep -qE "best_val_loss|best_model|best_epoch" "$ROOT/train.py" 2>/dev/null; then
    ok_at "Se rastrea el mejor modelo según la métrica de validación" \
          'best_val_loss|best_model|best_epoch' "$ROOT/train.py"
else
    warn "2.4 – No se ve seguimiento del mejor modelo según validation (revisar manualmente)"
fi

printf '    %bNota:%b La comprobación real de si “el modelo aprende” y “no overfittea”\n' "$YELLOW" "$RESET"
printf '    requiere ejecutar el entrenamiento y mirar las curvas / valores de loss.\n'
printf '    El script solo verifica que existan las herramientas para observarlo.\n'

echo

# -----------------------------------------------------------------------------
# 2.5  Saving – se guarda el modelo (topología + pesos)
# -----------------------------------------------------------------------------
info "2.5  Saving: ¿se guarda el modelo al final (topología + pesos)?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      grep -n "save\|dump\|weights\|architecture\|model_data" train.py utils.py network.py | head -25\n'
printf '      ls -la model*.npy model*.json saved_model* 2>/dev/null || true\n'
echo

SAVING_OK=1

if grep -qiE "save|dump|pickle|np\.save|json\.dump" "$ROOT/train.py" 2>/dev/null; then
    ok_at "train.py contiene lógica de guardado del modelo" \
          'save|dump|pickle|np\.save|json\.dump' "$ROOT/train.py"
else
    fail "No se encontró lógica de guardado (save/dump) en train.py"
    SAVING_OK=0
fi

# Comprobar que se guardan pesos y arquitectura
if grep -qE "weights|architecture|biases|topology" "$ROOT/train.py" 2>/dev/null; then
    ok_at "Se guardan pesos y/o arquitectura del modelo" \
          'weights|architecture|biases|topology' "$ROOT/train.py"
else
    fail "No se ve claramente el guardado de weights + architecture"
    SAVING_OK=0
fi

# Comprobar argumento --save
if grep -qE "\-\-save" "$ROOT/train.py" 2>/dev/null; then
    ok_at "train.py permite indicar la ruta de guardado con --save" \
          '\-\-save' "$ROOT/train.py"
else
    warn "2.5 – No se ve el argumento --save (puede tener un nombre fijo de fichero)"
fi

echo

# -----------------------------------------------------------------------------
# 2.6  Resumen del Paso 2
# -----------------------------------------------------------------------------
info "2.6  Resumen del Paso 2 (Training)"

echo
if [[ $DISPLAY_OK -eq 1 && $CURVES_OK -eq 1 && $METRICS_OK -eq 1 && $SAVING_OK -eq 1 ]]; then
    printf '%b✓ Paso 2 (Training) superado sin fallos críticos de código.%b\n' "$GREEN" "$RESET"
else
    printf '%b⚠  Hay puntos del Paso 2 que requieren revisión manual o corrección.%b\n' "$YELLOW" "$RESET"
fi

echo
printf '%bRecordatorio para el evaluador:%b\n' "$BOLD" "$RESET"
printf '  • Display  → mira que salgan métricas por época y las curvas al final.\n'
printf '  • Metrics  → confirma que hay loss/accuracy de train y de validation.\n'
printf '  • Model    → ejecuta un entrenamiento y observa si las métricas mejoran\n'
printf '               y si no hay overfitting evidente (val_loss disparado).\n'
printf '  • Saving   → verifica que se genera un fichero con topología + pesos.\n'

end_of_step "PASO 2 - Training"

# =============================================================================
# PASO 3 - Prediction (Loading, Metrics, Model performances)
# =============================================================================

section "PASO 3 - Prediction"

cat <<'EOF'
Esta sección evalúa la fase de predicción según la escala oficial:

  • Loading
      - El programa de predicción debe cargar los pesos y la topología
        de la red, así como el dataset sobre el que predecir.
      - El modelo entrenado se carga correctamente (no valores hard-coded).
      - El dataset de test/validación se carga correctamente.

  • Metrics
      - El programa de predicción evalúa el rendimiento con
        Binary Cross-Entropy (si no es así, no se puede evaluar
        la siguiente pregunta de la escala).

  • Model performances
      - Se ejecuta el flujo oficial (evaluation.py + data_training/test)
        y se mira el valor de Binary Cross-Entropy obtenido.
      - La nota de esta parte va de 0 a 5 según el umbral de BCE.

Parte de las comprobaciones son estáticas (código). La parte de
rendimiento requiere ejecutar predicción sobre un conjunto de test.
EOF

echo

# -----------------------------------------------------------------------------
# 3.1  Loading – carga del modelo y del dataset
# -----------------------------------------------------------------------------
info "3.1  Loading: ¿se cargan correctamente el modelo (topología + pesos) y el dataset?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      ls -la predict.py\n'
printf '      grep -n "load\|model\|weights\|architecture\|dataset" predict.py | head -30\n'
printf '      # Debe cargar architecture, weights, biases y el dataset de predicción\n'
echo

LOADING_OK=1

if [[ -f "$ROOT/predict.py" ]]; then
    ok "Existe el script predict.py"
else
    fail "No se encontró predict.py"
    LOADING_OK=0
fi

if [[ -f "$ROOT/predict.py" ]]; then
    if grep -qE "load.*model|load_model|weights|architecture" "$ROOT/predict.py"; then
        ok_at "predict.py carga el modelo (weights / architecture)" \
              'load.*model|load_model|weights|architecture' "$ROOT/predict.py"
    else
        fail "predict.py no parece cargar weights/architecture del modelo"
        LOADING_OK=0
    fi

    if grep -qE "dataset|load_data|preprocess|csv" "$ROOT/predict.py"; then
        ok_at "predict.py carga el dataset sobre el que se predicen" \
              'dataset|load_data|preprocess|\.csv' "$ROOT/predict.py"
    else
        fail "predict.py no parece cargar un dataset de predicción"
        LOADING_OK=0
    fi

    # Comprobar que no hay pesos hard-coded obvios
    if grep -qE "weights\s*=\s*\[\s*\[|hard.?coded" "$ROOT/predict.py"; then
        warn "3.1 – Posible hard-coding de pesos en predict.py (revisar manualmente)"
    else
        ok "No se detectan pesos hard-coded evidentes en predict.py"
    fi
fi

echo

# -----------------------------------------------------------------------------
# 3.2  Metrics – Binary Cross-Entropy en predicción
# -----------------------------------------------------------------------------
info "3.2  Metrics: ¿se evalúa con Binary Cross-Entropy en la predicción?"

printf '    %bComprobación manual recomendada:%b\n' "$YELLOW" "$RESET"
printf '      grep -n "binary_crossentropy\|cross.?entropy\|loss" predict.py loss.py | head -20\n'
printf '      # La escala exige Binary Cross-Entropy para poder puntuar el rendimiento\n'
echo

PRED_METRICS_OK=1

if [[ -f "$ROOT/predict.py" ]]; then
    if grep -qiE "binary_crossentropy|binary.?cross.?entropy" "$ROOT/predict.py"; then
        ok_at "predict.py usa Binary Cross-Entropy para evaluar" \
              'binary_crossentropy|binary.?cross.?entropy' "$ROOT/predict.py"
    else
        # Puede importarla desde loss.py
        if grep -qiE "from loss import|import loss|cross.?entropy" "$ROOT/predict.py"; then
            ok_at "predict.py importa/usa una función de cross-entropy (revisar que sea binary)" \
                  'from loss import|import loss|cross.?entropy' "$ROOT/predict.py"
        else
            fail "predict.py no parece calcular Binary Cross-Entropy"
            PRED_METRICS_OK=0
        fi
    fi
else
    fail "No existe predict.py → no se puede comprobar la métrica de predicción"
    PRED_METRICS_OK=0
fi

# Refuerzo: que loss.py tenga binary_crossentropy
if [[ -f "$ROOT/loss.py" ]]; then
    if grep -qiE "def\s+binary_crossentropy|binary_crossentropy" "$ROOT/loss.py"; then
        ok_at "loss.py define binary_crossentropy (fórmula del subject)" \
              'def\s+binary_crossentropy|binary_crossentropy' "$ROOT/loss.py"
    else
        warn "3.2 – No se ve def binary_crossentropy en loss.py"
    fi
fi

echo

# -----------------------------------------------------------------------------
# 3.3  Model performances – flujo oficial guiado paso a paso
# -----------------------------------------------------------------------------
info "3.3  Model performances: flujo oficial de evaluación de rendimiento"

echo
printf '%bEsta parte de la escala exige ejecutar el flujo oficial y mirar el Binary Cross-Entropy.%b\n' "$BOLD" "$RESET"
printf 'Se pueden entrenar hasta 3 modelos y quedarse con el mejor resultado.\n'
echo

# --- Función auxiliar: pausa entre sub-pasos del flujo oficial ---
pause_step() {
    if [[ "$INTERACTIVE" -eq 1 && "$PAUSE" -eq 1 ]]; then
        printf '\n%bPulsa INTRO para continuar...%b' "$YELLOW" "$RESET"
        read -r
        echo
    fi
}

# ----- Paso 1 -----
printf '%b▶ 1. Descargar evaluation.py de los recursos del proyecto.%b\n' "$CYAN" "$RESET"
printf '   El fichero está en los attachments de la escala (intranet 42).\n'
printf '   %bEjecutar:%b\n' "$YELLOW" "$RESET"
printf '      # Si ya lo tienes en el directorio del proyecto, no hace falta nada.\n'
printf '      # Si no, descárgalo desde la página de evaluación y colócalo aquí:\n'
printf '      ls -la evaluation.py\n'
printf '   %bResultado esperado:%b debe existir el fichero evaluation.py en el directorio del proyecto.\n' "$GREEN" "$RESET"
pause_step

# ----- Paso 2 -----
printf '%b▶ 2. Ejecutar evaluation.py%b\n' "$CYAN" "$RESET"
printf '   (descarga el dataset, lo baraja y lo parte en train/test).\n'
printf '   %bEjecutar:%b\n' "$YELLOW" "$RESET"
printf '      python3 evaluation.py\n'
printf '   %bResultado esperado:%b se generan (o actualizan) los ficheros\n' "$GREEN" "$RESET"
printf '      data_training.csv y data_test.csv (o nombres similares que indique el script).\n'
pause_step

# ----- Paso 3 -----
printf '%b▶ 3. Entrenar un modelo nuevo con data_training.csv%b\n' "$CYAN" "$RESET"
printf '   %bEjecutar:%b\n' "$YELLOW" "$RESET"
printf '      python3 train.py --dataset data_training.csv --epochs 100 --verbose\n'
echo
printf '   %bAjustes opcionales más útiles:%b\n' "$YELLOW" "$RESET"
printf '      --layers 24 24          Neuronas por cada capa oculta (default: 24 24).\n'
printf '                             Más capas/neuronas = más capacidad, más riesgo de overfitting.\n'
printf '      --learning_rate 0.05   Tamaño del paso al actualizar pesos.\n'
printf '                             Muy alto → inestable; muy bajo → aprende lento.\n'
printf '      --batch_size 16        Cuántas muestras se usan en cada actualización de pesos.\n'
printf '      --activation relu      Función de activación de las capas ocultas\n'
printf '                             (sigmoid, tanh, relu, leaky_relu).\n'
printf '      --optimizer adam       Optimizador: sgd (baseline), sgd_momentum, rmsprop, adam.\n'
printf '                             Adam suele converger más rápido.\n'
printf '      --early_stopping       Para el entrenamiento si la validación deja de mejorar.\n'
printf '      --patience 10          Épocas sin mejora antes de parar (con early_stopping).\n'
printf '      --dropout 0.2          Apaga neuronas al azar durante el entrenamiento (regularización).\n'
printf '      --save model_weights.npy   Fichero donde se guarda el modelo\n'
printf '                                 (default del repo: model_weights.npy).\n'
printf '      --seed 42              Semilla para reproducibilidad.\n'
echo
printf '   %bResultado esperado:%b\n' "$GREEN" "$RESET"
printf '      - Se muestran métricas por época (loss, val_loss, acc, val_acc).\n'
printf '      - Al final se guarda el modelo en model_weights.npy\n'
printf '        (o el nombre que hayas pasado con --save).\n'
pause_step

# ----- Paso 4 -----
printf '%b▶ 4. Predecir sobre data_test.csv con ese modelo%b\n' "$CYAN" "$RESET"
printf '   %bEjecutar:%b\n' "$YELLOW" "$RESET"
printf '      python3 predict.py --model model_weights.npy --dataset data_test.csv\n'
echo
printf '   %bNombre del modelo en este repositorio:%b\n' "$YELLOW" "$RESET"
printf '      Por defecto el proyecto guarda y carga:  %bmodel_weights.npy%b\n' "$BOLD" "$RESET"
printf '      (definido en config.py como MODEL_FILE = '\''model_weights.npy'\'').\n'
printf '      Solo cambia --model si en el entrenamiento usaste otro nombre con --save.\n'
echo
printf '   %bResultado esperado:%b\n' "$GREEN" "$RESET"
printf '      - Se cargan el modelo (model_weights.npy) y el dataset de test.\n'
printf '      - Se muestran métricas de evaluación, entre ellas el Binary Cross-Entropy.\n'
pause_step

# ----- Paso 5 -----
printf '%b▶ 5. Mirar el Binary Cross-Entropy obtenido en test%b\n' "$CYAN" "$RESET"
printf '   Observa en la salida de predict.py la línea que indica el valor de\n'
printf '   Binary Cross-Entropy (BCE) sobre el conjunto de test.\n'
echo
printf '   %b¿Qué significa ese valor? (explicación sencilla)%b\n' "$BOLD" "$RESET"
printf '      El Binary Cross-Entropy mide cómo de “equivocada” está la red en promedio.\n'
printf '      - Si vale cerca de 0 → la red está muy segura y casi siempre acierta.\n'
printf '      - Si vale cerca de 0.7 → es casi como tirar una moneda (no ha aprendido bien).\n'
printf '      - Cuanto MÁS BAJO sea el número, MEJOR es el modelo.\n'
echo
printf '   %bValoración según la escala oficial (BCE en test):%b\n' "$BOLD" "$RESET"
printf '        BCE > 0.35          →  0 puntos\n'
printf '        0.35 ≥ BCE > 0.25   →  1 punto\n'
printf '        0.25 ≥ BCE > 0.18   →  2 puntos\n'
printf '        0.18 ≥ BCE > 0.13   →  3 puntos\n'
printf '        0.13 ≥ BCE > 0.08   →  4 puntos\n'
printf '        BCE ≤ 0.08          →  5 puntos (excelente)\n'
echo
printf '   Puedes repetir los pasos 3 y 4 hasta 3 veces (con distintas semillas o\n'
printf '   hiperparámetros) y quedarte con el mejor BCE para puntuar esta casilla.\n'
pause_step

# Comprobaciones de soporte (no ejecutan el entrenamiento completo aquí)
if [[ -f "$ROOT/predict.py" && -f "$ROOT/train.py" ]]; then
    ok "Existen train.py y predict.py para poder ejecutar el flujo de rendimiento"
else
    fail "Faltan train.py y/o predict.py → no se puede completar el flujo de rendimiento"
fi

# Ver si hay algún resultado previo o historial
if ls "$ROOT"/model*.npy "$ROOT"/model*.json "$ROOT"/saved_model* 2>/dev/null | head -1 >/dev/null; then
    ok "Se encontraron ficheros de modelo guardados (útiles para predecir)"
    ls -1 "$ROOT"/model*.npy "$ROOT"/model*.json "$ROOT"/saved_model* 2>/dev/null | head -5 | while read -r f; do
        info "    → $(basename "$f")"
    done
else
    info "3.3 – No hay modelos guardados todavía (normal si aún no se ha entrenado)"
    info "      Entrena antes con:  python3 train.py --dataset data_training.csv --verbose"
fi

echo

# -----------------------------------------------------------------------------
# 3.4  Resumen del Paso 3
# -----------------------------------------------------------------------------
info "3.4  Resumen del Paso 3 (Prediction)"

echo
if [[ $LOADING_OK -eq 1 && $PRED_METRICS_OK -eq 1 ]]; then
    printf '%b✓ Paso 3 (Prediction) superado sin fallos críticos de código.%b\n' "$GREEN" "$RESET"
else
    printf '%b⚠  Hay puntos del Paso 3 que requieren revisión manual o corrección.%b\n' "$YELLOW" "$RESET"
fi

echo
printf '%bRecordatorio para el evaluador:%b\n' "$BOLD" "$RESET"
printf '  • Loading  → verificar que se cargan topología + pesos + dataset (sin hard-code).\n'
printf '  • Metrics  → confirmar que se usa Binary Cross-Entropy en la predicción.\n'
printf '  • Performances → ejecutar el flujo oficial (evaluation.py + train + predict)\n'
printf '                   y puntuar 0-5 según el BCE obtenido en test.\n'

end_of_step "PASO 3 - Prediction"

# =============================================================================
# PASO 4 - Algorithms understanding
# =============================================================================

section "PASO 4 - Algorithms understanding"

cat <<'EOF'
La escala pide evaluar si el estudiante ENTIENDE los algoritmos
(no solo si el código funciona). Cada bloque se puntúa de 0 a 5.

  • Feedforward
  • Gradient Descent
  • Backpropagation
  • Overfitting

A continuación tienes una explicación sencilla de cada uno para
repasar con el estudiante. Tras cada algoritmo hay una pausa.
EOF

echo

# -----------------------------------------------------------------------------
# 4.1  Feedforward
# -----------------------------------------------------------------------------
info "4.1  Feedforward"

printf '%b¿Qué es?%b\n' "$BOLD" "$RESET"
printf '  Es el camino de ida: los datos entran y atraviesan la red hasta la salida.\n'
printf '  No se aprende nada aquí; solo se calcula una predicción.\n'
echo
printf '%bEjemplo cotidiano:%b\n' "$BOLD" "$RESET"
printf '  Como una cadena de personas que pasan un mensaje.\n'
printf '  Cada persona recibe números, los combina (pesos + bias), aplica su “filtro”\n'
printf '  (función de activación) y pasa el resultado a la siguiente.\n'
printf '  La última persona dice: “creo que es maligno con un 80 %%”.\n'
echo
printf '%bEn la red:%b\n' "$BOLD" "$RESET"
printf '  Entrada → Capa oculta 1 → Capa oculta 2 → Salida (Softmax) → probabilidades\n'
echo
printf '%bFórmula (por capa):%b\n' "$BOLD" "$RESET"
printf '  z = W · a_anterior + b\n'
printf '  a = activación(z)          (ReLU, sigmoid, tanh, ...)\n'
printf '  En la salida:  a = softmax(z)  →  probabilidades que suman 1\n'
echo
printf '%bQué debe ser capaz de explicar el estudiante:%b\n' "$BOLD" "$RESET"
printf '  • Orden del cálculo (de izquierda a derecha).\n'
printf '  • Qué son z, W, b y a.\n'
printf '  • Para qué sirve la activación y por qué Softmax al final.\n'
echo
printf '  %bPuntuación escala: 0 (no entiende) … 5 (excelente)%b\n' "$YELLOW" "$RESET"
pause_step

# -----------------------------------------------------------------------------
# 4.2  Gradient Descent
# -----------------------------------------------------------------------------
info "4.2  Gradient Descent"

printf '%b¿Qué es?%b\n' "$BOLD" "$RESET"
printf '  Es el método para MEJORAR la red: mover los pesos un poquito en la\n'
printf '  dirección que reduce el error.\n'
echo
printf '%bEjemplo cotidiano:%b\n' "$BOLD" "$RESET"
printf '  Estás vendado en una montaña y quieres bajar al valle (error mínimo).\n'
printf '  Con el pie notas hacia dónde baja el terreno (el gradiente).\n'
printf '  Das un paso en esa dirección. El tamaño del paso es el learning rate.\n'
printf '  - Paso muy grande → te pasas y subes por el otro lado.\n'
printf '  - Paso muy pequeño → tardas una eternidad.\n'
echo
printf '%bIdea clave:%b\n' "$BOLD" "$RESET"
printf '  gradiente = “en qué dirección crece el error”.\n'
printf '  Por eso se RESTA el gradiente: vamos en dirección contraria al crecimiento.\n'
echo
printf '%bFórmula:%b\n' "$BOLD" "$RESET"
printf '  θ_nuevo = θ_viejo − α · ∇E\n'
printf '  donde:\n'
printf '    θ  = un peso o un bias\n'
printf '    α  = learning rate (tamaño del paso)\n'
printf '    ∇E = gradiente del error respecto a ese parámetro\n'
echo
printf '%bAdam (bonus, si sale en la charla):%b\n' "$BOLD" "$RESET"
printf '  Misma idea, pero más listo: recuerda la dirección anterior (momentum)\n'
printf '  y adapta el tamaño del paso según lo “abrupto” del terreno.\n'
echo
printf '%bQué debe ser capaz de explicar el estudiante:%b\n' "$BOLD" "$RESET"
printf '  • Por qué se resta el gradiente.\n'
printf '  • Qué es el learning rate y qué pasa si es muy grande o muy pequeño.\n'
printf '  • Diferencia intuitiva entre SGD y Adam (si lo ha implementado).\n'
echo
printf '  %bPuntuación escala: 0 (no entiende) … 5 (excelente)%b\n' "$YELLOW" "$RESET"
pause_step

# -----------------------------------------------------------------------------
# 4.3  Backpropagation
# -----------------------------------------------------------------------------
info "4.3  Backpropagation"

printf '%b¿Qué es?%b\n' "$BOLD" "$RESET"
printf '  Es el camino de vuelta: después de ver el error en la salida,\n'
printf '  repartimos la “culpa” hacia atrás, capa por capa, hasta la entrada.\n'
printf '  Así sabemos cuánto debe cambiar cada peso.\n'
echo
printf '%bEjemplo cotidiano:%b\n' "$BOLD" "$RESET"
printf '  Un equipo de cocina prepara un plato y el cliente dice “está demasiado salado”.\n'
printf '  - Primero miramos el resultado final (el error).\n'
printf '  - Preguntamos al último cocinero: “¿quién te pasó el salero?”.\n'
printf '  - Él señala al anterior, y así vamos hacia atrás.\n'
printf '  - Cada uno ajusta un poco su forma de cocinar para la próxima vez.\n'
echo
printf '%bPasos resumidos:%b\n' "$BOLD" "$RESET"
printf '  1. Calcular el error en la salida (con Softmax + Cross-Entropy se simplifica).\n'
printf '  2. Propagar ese error hacia atrás (regla de la cadena).\n'
printf '  3. Calcular el gradiente de cada peso y cada bias.\n'
printf '  4. Actualizar parámetros con Gradient Descent / Adam.\n'
echo
printf '%bFórmulas útiles:%b\n' "$BOLD" "$RESET"
printf '  Salida (Softmax + Cross-Entropy):\n'
printf '    δ_salida = predicción − realidad          (ŷ − y)\n'
echo
printf '  Capas ocultas:\n'
printf '    δ_capa = (W_siguienteᵀ · δ_siguiente) ⊙ activación'\''(z)\n'
printf '    (⊙ = producto elemento a elemento)\n'
echo
printf '  Gradientes:\n'
printf '    ∂E/∂W = δ · (activación_anterior)ᵀ\n'
printf '    ∂E/∂b = δ\n'
echo
printf '%bQué debe ser capaz de explicar el estudiante:%b\n' "$BOLD" "$RESET"
printf '  • Por qué hay que ir hacia atrás (el error solo se ve al final).\n'
printf '  • Qué es δ (delta): cantidad de culpa de cada neurona.\n'
printf '  • Cómo se usa la regla de la cadena (sin necesidad de demostrar todo).\n'
printf '  • La simplificación ŷ − y cuando hay Softmax + Cross-Entropy.\n'
echo
printf '  %bPuntuación escala: 0 (no entiende) … 5 (excelente)%b\n' "$YELLOW" "$RESET"
pause_step

# -----------------------------------------------------------------------------
# 4.4  Overfitting
# -----------------------------------------------------------------------------
info "4.4  Overfitting"

printf '%b¿Qué es?%b\n' "$BOLD" "$RESET"
printf '  Es cuando la red se aprende de memoria el conjunto de entrenamiento\n'
printf '  y luego falla con datos nuevos (validación / test).\n'
echo
printf '%bEjemplo cotidiano:%b\n' "$BOLD" "$RESET"
printf '  Un estudiante memoriza las preguntas exactas de los exámenes de años\n'
printf '  anteriores. En esos exámenes saca un 10. El día del examen real,\n'
printf '  con preguntas nuevas, suspende.\n'
printf '  Ha memorizado, no ha generalizado.\n'
echo
printf '%bCómo se detecta:%b\n' "$BOLD" "$RESET"
printf '  - Loss / accuracy de entrenamiento siguen mejorando.\n'
printf '  - Loss de validación deja de mejorar o EMPIEZA A EMPEORAR.\n'
printf '  Esa divergencia es la señal clásica de overfitting.\n'
echo
printf '%bCómo se combate (en este proyecto):%b\n' "$BOLD" "$RESET"
printf '  • Conjunto de validación (20 %): vigilar el rendimiento en datos no vistos.\n'
printf '  • Early stopping: parar cuando val_loss deja de mejorar (patience).\n'
printf '  • Dropout (bonus): apagar neuronas al azar para que no dependa de pocas.\n'
printf '  • Arquitectura moderada (no enormes capas con pocos datos).\n'
printf '  • Más datos / regularización (si se dispusiera de ellos).\n'
echo
printf '%bQué debe ser capaz de explicar el estudiante:%b\n' "$BOLD" "$RESET"
printf '  • Diferencia entre “aprender el patrón” y “memorizar los ejemplos”.\n'
printf '  • Cómo se ve en las curvas train vs validation.\n'
printf '  • Qué medidas ha usado en su código para evitarlo.\n'
echo
printf '  %bPuntuación escala: 0 (no entiende) … 5 (excelente)%b\n' "$YELLOW" "$RESET"
pause_step

# -----------------------------------------------------------------------------
# 4.5  Resumen del Paso 4
# -----------------------------------------------------------------------------
info "4.5  Resumen del Paso 4 (Algorithms understanding)"

echo
printf '%bRecordatorio para el evaluador:%b\n' "$BOLD" "$RESET"
printf '  Pregunta al estudiante cada bloque y puntúa 0–5 según la claridad\n'
printf '  y profundidad de su explicación (con el código delante si hace falta).\n'
printf '  No hace falta que recite fórmulas de memoria; sí que demuestre\n'
printf '  que entiende el “qué” y el “por qué”.\n'
echo
printf '  Guía rápida de puntuación:\n'
printf '    0–1  No sabe o confunde conceptos básicos\n'
printf '    2–3  Idea general correcta, fallos en detalles\n'
printf '    4    Buena explicación con ejemplo o fórmula\n'
printf '    5    Clara, completa y con intuición sólida\n'

end_of_step "PASO 4 - Algorithms understanding"

# =============================================================================
# PASO 5 - Bonus
# =============================================================================

section "PASO 5 - Bonus"

cat <<'EOF'
IMPORTANTE (texto de la escala):
  La sección Bonus solo se evalúa si la parte obligatoria está
  perfectamente hecha.

  • Se atribuye UN PUNTO por cada bonus real.
  • Un bonus debe suponer un mínimo de inversión de tiempo.
    Cosas que se añaden en 5 minutos no merecen punto.
  • Puntuación final de esta casilla: 0 a 5.

A continuación se listan los bonus detectables en este repositorio.
Cada uno muestra dónde está en el código para que el evaluador
pueda comprobarlo con el estudiante.
EOF

echo

BONUS_COUNT=0

# --- Helper: registra un bonus encontrado ---
bonus_found() {
    local title="$1"
    local pattern="$2"
    shift 2
    local location
    location=$(find_lines "$pattern" "$@")
    BONUS_COUNT=$((BONUS_COUNT + 1))
    if [[ -n "$location" ]]; then
        ok "Bonus $BONUS_COUNT: $title  →  $location"
    else
        ok "Bonus $BONUS_COUNT: $title"
    fi
}

bonus_missing() {
    local title="$1"
    printf '%b[----]%b No detectado: %s\n' "$YELLOW" "$RESET" "$title"
    log "[----] No detectado: $title"
}

# -----------------------------------------------------------------------------
# 5.1  Optimizadores avanzados (Adam, RMSprop, Momentum)
# -----------------------------------------------------------------------------
info "5.1  Optimizadores avanzados"

printf '    %bComprobación manual:%b  grep -n "adam\\|rmsprop\\|momentum" optimizers.py train.py\n' "$YELLOW" "$RESET"
echo

if [[ -f "$ROOT/optimizers.py" ]] && grep -qiE "class Adam|def.*adam|adam" "$ROOT/optimizers.py" 2>/dev/null; then
    bonus_found "Adam optimizer" 'class Adam|adam' "$ROOT/optimizers.py" "$ROOT/train.py"
else
    if grep -qiE "adam" "$ROOT/train.py" "$ROOT/network.py" 2>/dev/null; then
        bonus_found "Adam optimizer (referenciado en train/network)" 'adam' "$ROOT/train.py" "$ROOT/network.py"
    else
        bonus_missing "Adam optimizer"
    fi
fi

if [[ -f "$ROOT/optimizers.py" ]] && grep -qiE "rmsprop|RMSprop" "$ROOT/optimizers.py" 2>/dev/null; then
    bonus_found "RMSprop optimizer" 'rmsprop|RMSprop' "$ROOT/optimizers.py"
else
    bonus_missing "RMSprop optimizer"
fi

if [[ -f "$ROOT/optimizers.py" ]] && grep -qiE "momentum|SGDMomentum" "$ROOT/optimizers.py" 2>/dev/null; then
    bonus_found "SGD con Momentum" 'momentum|SGDMomentum' "$ROOT/optimizers.py"
else
    bonus_missing "SGD con Momentum"
fi

echo

# -----------------------------------------------------------------------------
# 5.2  Early stopping
# -----------------------------------------------------------------------------
info "5.2  Early stopping"

printf '    %bComprobación manual:%b  grep -n "early_stopping\\|patience" train.py\n' "$YELLOW" "$RESET"
echo

if grep -qiE "early_stopping|patience" "$ROOT/train.py" 2>/dev/null; then
    bonus_found "Early stopping (patience)" 'early_stopping|patience' "$ROOT/train.py"
else
    bonus_missing "Early stopping"
fi

echo

# -----------------------------------------------------------------------------
# 5.3  Dropout
# -----------------------------------------------------------------------------
info "5.3  Dropout"

printf '    %bComprobación manual:%b  grep -n "dropout" train.py layers.py network.py\n' "$YELLOW" "$RESET"
echo

if grep -qiE "dropout" "$ROOT/train.py" "$ROOT/layers.py" "$ROOT/network.py" 2>/dev/null; then
    bonus_found "Dropout regularization" 'dropout' "$ROOT/train.py" "$ROOT/layers.py" "$ROOT/network.py"
else
    bonus_missing "Dropout"
fi

echo

# -----------------------------------------------------------------------------
# 5.4  Múltiples activaciones
# -----------------------------------------------------------------------------
info "5.4  Múltiples funciones de activación"

printf '    %bComprobación manual:%b  grep -n "def relu\\|def tanh\\|def sigmoid\\|leaky" activation.py\n' "$YELLOW" "$RESET"
echo

ACT_COUNT=0
[[ -f "$ROOT/activation.py" ]] && grep -qE "def\s+sigmoid" "$ROOT/activation.py" && ACT_COUNT=$((ACT_COUNT+1))
[[ -f "$ROOT/activation.py" ]] && grep -qE "def\s+tanh" "$ROOT/activation.py" && ACT_COUNT=$((ACT_COUNT+1))
[[ -f "$ROOT/activation.py" ]] && grep -qE "def\s+relu" "$ROOT/activation.py" && ACT_COUNT=$((ACT_COUNT+1))
[[ -f "$ROOT/activation.py" ]] && grep -qE "def\s+leaky_relu|leaky" "$ROOT/activation.py" && ACT_COUNT=$((ACT_COUNT+1))

if [[ "$ACT_COUNT" -ge 3 ]]; then
    bonus_found "Múltiples activaciones ($ACT_COUNT: sigmoid/tanh/relu/leaky...)" \
                'def\s+sigmoid|def\s+tanh|def\s+relu|def\s+leaky' "$ROOT/activation.py"
else
    bonus_missing "Múltiples activaciones (se encontraron $ACT_COUNT)"
fi

echo

# -----------------------------------------------------------------------------
# 5.5  Inicialización de pesos (He / Xavier)
# -----------------------------------------------------------------------------
info "5.5  Inicialización de pesos (He / Xavier)"

printf '    %bComprobación manual:%b  grep -n "he\\|xavier\\|He\\|Xavier" initialization.py train.py\n' "$YELLOW" "$RESET"
echo

if [[ -f "$ROOT/initialization.py" ]] && grep -qiE "he|xavier" "$ROOT/initialization.py" 2>/dev/null; then
    bonus_found "Inicialización He / Xavier" 'he|xavier|He|Xavier' "$ROOT/initialization.py" "$ROOT/train.py"
elif grep -qiE "weight_init|he|xavier" "$ROOT/train.py" 2>/dev/null; then
    bonus_found "Inicialización He / Xavier (vía train.py)" 'weight_init|he|xavier' "$ROOT/train.py"
else
    bonus_missing "Inicialización He / Xavier"
fi

echo

# -----------------------------------------------------------------------------
# 5.6  Arquitectura configurable por CLI
# -----------------------------------------------------------------------------
info "5.6  Arquitectura configurable (--layers y más)"

printf '    %bComprobación manual:%b  grep -n "\\-\\-layers\\|add_argument" train.py | head -20\n' "$YELLOW" "$RESET"
echo

if grep -qE "\-\-layers" "$ROOT/train.py" 2>/dev/null; then
    bonus_found "Arquitectura configurable por argumentos CLI" '\-\-layers' "$ROOT/train.py"
else
    bonus_missing "Arquitectura configurable por CLI"
fi

echo

# -----------------------------------------------------------------------------
# 5.7  Métricas adicionales y visualización
# -----------------------------------------------------------------------------
info "5.7  Métricas adicionales y curvas de aprendizaje"

printf '    %bComprobación manual:%b  ls metrics.py visualize.py; grep -n "precision\\|recall\\|f1\\|plot" metrics.py visualize.py\n' "$YELLOW" "$RESET"
echo

if [[ -f "$ROOT/metrics.py" ]] && grep -qiE "precision|recall|f1|F1|confusion" "$ROOT/metrics.py" 2>/dev/null; then
    bonus_found "Métricas extra (precision / recall / F1 / matriz)" \
                'precision|recall|f1|F1|confusion' "$ROOT/metrics.py"
else
    bonus_missing "Métricas extra (precision, recall, F1...)"
fi

if [[ -f "$ROOT/visualize.py" ]] && grep -qiE "plot|matplotlib|curve" "$ROOT/visualize.py" 2>/dev/null; then
    bonus_found "Curvas de aprendizaje (visualize.py)" \
                'plot|matplotlib|curve' "$ROOT/visualize.py"
else
    bonus_missing "Curvas de aprendizaje"
fi

echo

# -----------------------------------------------------------------------------
# 5.8  Mini-batch training
# -----------------------------------------------------------------------------
info "5.8  Entrenamiento por mini-batches"

printf '    %bComprobación manual:%b  grep -n "batch" train.py | head -15\n' "$YELLOW" "$RESET"
echo

if grep -qiE "batch_size|create_batches|mini.?batch" "$ROOT/train.py" 2>/dev/null; then
    bonus_found "Mini-batch gradient descent" 'batch_size|create_batches|mini.?batch' "$ROOT/train.py"
else
    bonus_missing "Mini-batch training"
fi

echo

# -----------------------------------------------------------------------------
# 5.9  Resumen y puntuación Bonus
# -----------------------------------------------------------------------------
info "5.9  Resumen del Paso 5 (Bonus)"

echo
printf '%bBonus detectados automáticamente: %d%b\n' "$BOLD" "$BONUS_COUNT" "$RESET"
echo
printf '%bCriterio de la escala:%b\n' "$BOLD" "$RESET"
printf '  • Solo puntuar Bonus si el mandatory está perfecto.\n'
printf '  • 1 punto por cada bonus que implique trabajo real (no trivial).\n'
printf '  • Máximo de esta casilla: 5.\n'
echo
printf '  Sugerencia de puntuación orientativa:\n'
printf '    0 bonus útiles     →  0\n'
printf '    1–2 bonus sólidos  →  1–2\n'
printf '    3–4 bonus sólidos  →  3–4\n'
printf '    5+ bonus sólidos   →  5\n'
echo
printf '  %bLista orientativa de lo encontrado en este repo:%b\n' "$YELLOW" "$RESET"
printf '    Adam / RMSprop / Momentum, Early stopping, Dropout,\n'
printf '    Varias activaciones, He/Xavier, CLI configurable,\n'
printf '    Métricas extra, Curvas, Mini-batches.\n'
echo
printf '  El evaluador decide cuántos merecen punto real según el esfuerzo\n'
printf '  demostrado y la calidad de la implementación/defensa.\n'

end_of_step "PASO 5 - Bonus"

# =============================================================================
# FIN DE LA EVALUACIÓN
# =============================================================================

section "Fin de la evaluación"

printf 'Secciones completadas:\n'
printf '  ✓ PASO 0 - Comprobaciones previas (Guidelines)\n'
printf '  ✓ PASO 1 - Implementation (Modularidad, Dataset split, Softmax)\n'
printf '  ✓ PASO 2 - Training (Display, Metrics, Model, Saving)\n'
printf '  ✓ PASO 3 - Prediction (Loading, Metrics, Model performances)\n'
printf '  ✓ PASO 4 - Algorithms understanding\n'
printf '  ✓ PASO 5 - Bonus\n'
echo
printf '%bLa escala también pide:%b\n' "$BOLD" "$RESET"
printf '  • Marcar el flag correspondiente (Ok / Outstanding / Empty / Incomplete / ...)\n'
printf '  • Dejar un comentario final (máx. 2048 caracteres)\n'
echo

printf '%bEstado actual:%b  Total=%d  OK=%d  FAIL=%d  Avisos=%d  Bonus detectados=%d\n' \
    "$BOLD" "$RESET" "$TOTAL" "$PASSED" "$FAILED" "$WARNINGS" "$BONUS_COUNT"

if [[ "$WARNINGS" -gt 0 ]]; then
    printf '\n%bNota:%b Hay %d aviso(s). Detalle:\n' "$YELLOW" "$RESET" "$WARNINGS"
    if [[ -f "$LOG" ]]; then
        grep '\[AVISO\]' "$LOG" 2>/dev/null | while IFS= read -r line; do
            printf '  %b%s%b\n' "$YELLOW" "$line" "$RESET"
        done
    fi
    printf '\nRevisa las líneas [AVISO] de la salida o el log para más contexto.\n'
fi

printf '\nLog completo disponible en:\n  %s\n\n' "$LOG"

if [[ "$KEEP_ARTIFACTS" -eq 0 ]]; then
    info "Limpiando directorio de artefactos (--clean activado)..."
    rm -rf "$ARTIFACTS"
fi

printf '%bEvaluación asistida finalizada.%b\n\n' "$GREEN" "$RESET"
exit 0
