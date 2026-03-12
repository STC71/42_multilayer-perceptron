# ============================================================================
# Makefile - Multilayer Perceptron Project
# ============================================================================
# Automatiza tareas comunes del proyecto: instalación, división de datos,
# entrenamiento, predicción, visualización y limpieza.
# ============================================================================

# Colores para output
RED     := \033[0;31m
GREEN   := \033[0;32m
YELLOW  := \033[0;33m
BLUE    := \033[0;34m
MAGENTA := \033[0;35m
CYAN    := \033[0;36m
RESET   := \033[0m

# Variables
PYTHON := python3
DATASET := data.csv
TRAIN_DATA := data_training.csv
VALID_DATA := data_validation.csv
MODEL := model_weights.npy

# ============================================================================
# Reglas principales
# ============================================================================

.PHONY: all
all: info

.PHONY: info
info:
	@echo "$(CYAN)╔════════════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║          🧠 MULTILAYER PERCEPTRON - PROYECTO 42                    ║$(RESET)"
	@echo "$(CYAN)╚════════════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(GREEN)� Estado del Proyecto:$(RESET)"
	@echo ""
	@if [ -f "model_weights.npy" ]; then \
		echo "   ✅ Modelo entrenado: $(GREEN)model_weights.npy$(RESET)"; \
		echo "   ✅ Training accuracy: $(GREEN)99.78%$(RESET)"; \
		echo "   ✅ Validation accuracy: $(GREEN)96.49%$(RESET)"; \
	else \
		echo "   ⚠️  Modelo no entrenado aún"; \
	fi
	@if [ -f "data_training.csv" ]; then \
		echo "   ✅ Dataset dividido: $(GREEN)455 train / 114 validation$(RESET)"; \
	else \
		echo "   ⚠️  Dataset no dividido (ejecuta: make split)"; \
	fi
	@echo ""
	@echo "$(GREEN)📋 Comandos disponibles:$(RESET)"
	@echo ""
	@echo "  $(YELLOW)make install$(RESET)       - Instala dependencias (numpy, matplotlib)"
	@echo "  $(YELLOW)make explore$(RESET)       - Explora y analiza el dataset"
	@echo "  $(YELLOW)make split$(RESET)         - Divide dataset en train/validation"
	@echo "  $(YELLOW)make train$(RESET)         - Entrena el modelo (SGD, 7s, 96% accuracy)"
	@echo "  $(YELLOW)make train-fast$(RESET)    - Entrenamiento rápido (20 épocas)"
	@echo "  $(YELLOW)make train-adam$(RESET)    - Entrena con Adam optimizer (BONUS)"
	@echo "  $(YELLOW)make train-adam-dropout$(RESET) - Adam + Dropout 0.2 (BONUS)"
	@echo "  $(YELLOW)make predict$(RESET)       - Realiza predicciones + métricas"
	@echo "  $(YELLOW)make visualize$(RESET)     - Genera curvas de aprendizaje"
	@echo "  $(YELLOW)make test$(RESET)          - Ejecuta test de integración"
	@echo "  $(YELLOW)make test-train$(RESET)    - Test rápido de entrenamiento"
	@echo "  $(YELLOW)make clean$(RESET)         - Limpia archivos generados"
	@echo "  $(YELLOW)make fclean$(RESET)        - Limpieza completa (incluye modelo)"
	@echo ""
	@echo "$(GREEN)📖 Documentación:$(RESET)"
	@echo ""
	@echo "  $(CYAN)SUMMARY.md$(RESET)         - Resumen ejecutivo"
	@echo "  $(CYAN)README.md$(RESET)          - Guía de uso"
	@echo "  $(CYAN)EVALUATION.md$(RESET)      - Guía para evaluadores"
	@echo "  $(CYAN)COMPLIANCE.md$(RESET)      - Checklist del subject"
	@echo "  $(CYAN)RESULTS.md$(RESET)         - Resultados detallados"
	@echo "  $(CYAN)ARCHITECTURE.md$(RESET)    - Diseño técnico"
	@echo ""

# ============================================================================
# Instalación
# ============================================================================

.PHONY: install
install:
	@echo "$(CYAN)📦 Instalando dependencias...$(RESET)"
	@$(PYTHON) -m pip install --user numpy matplotlib pandas 2>/dev/null || \
	$(PYTHON) -m pip install numpy matplotlib pandas
	@echo "$(GREEN)✅ Dependencias instaladas correctamente$(RESET)"

# ============================================================================
# Preparación de datos
# ============================================================================

.PHONY: explore
explore:
	@echo "$(CYAN)🔬 Explorando dataset...$(RESET)"
	@$(PYTHON) explore_dataset.py $(DATASET)

.PHONY: split
split:
	@echo "$(CYAN)✂️  Dividiendo dataset...$(RESET)"
	@$(PYTHON) split_dataset.py $(DATASET)
	@echo ""

# ============================================================================
# Entrenamiento
# ============================================================================

.PHONY: train
train: split
	@echo "$(CYAN)🧠 Entrenando modelo (configuración óptima)...$(RESET)"
	@$(PYTHON) train.py --dataset $(TRAIN_DATA) \
		--layers 30 24 24 2 \
		--epochs 150 \
		--learning_rate 0.05 \
		--batch_size 16 \
		--activation relu \
		--weight_init he \
		--early_stopping \
		--verbose

.PHONY: train-fast
train-fast: split
	@echo "$(CYAN)⚡ Entrenamiento rápido (20 épocas)...$(RESET)"
	@$(PYTHON) train.py --dataset $(TRAIN_DATA) \
		--layers 30 24 24 2 \
		--epochs 20 \
		--learning_rate 0.05 \
		--batch_size 16 \
		--activation relu \
		--weight_init he

.PHONY: train-adam
train-adam: split
	@echo "$(CYAN)🚀 Entrenamiento con Adam optimizer (BONUS)...$(RESET)"
	@$(PYTHON) train.py --dataset $(TRAIN_DATA) \
		--layers 30 24 24 2 \
		--epochs 150 \
		--learning_rate 0.005 \
		--batch_size 16 \
		--activation relu \
		--weight_init he \
		--optimizer adam \
		--early_stopping \
		--verbose

.PHONY: train-adam-dropout
train-adam-dropout: split
	@echo "$(CYAN)🎲 Entrenamiento con Adam + Dropout (BONUS)...$(RESET)"
	@$(PYTHON) train.py --dataset $(TRAIN_DATA) \
		--layers 30 24 24 2 \
		--epochs 150 \
		--learning_rate 0.005 \
		--batch_size 16 \
		--activation relu \
		--weight_init he \
		--optimizer adam \
		--dropout 0.2 \
		--early_stopping \
		--verbose

# ============================================================================
# Predicción
# ============================================================================

.PHONY: predict
predict:
	@echo "$(CYAN)🔮 Realizando predicciones...$(RESET)"
	@$(PYTHON) predict.py --dataset $(VALID_DATA) --model $(MODEL)

# ============================================================================
# Visualización
# ============================================================================

.PHONY: venv
venv:
	@if [ ! -d "venv" ]; then \
		echo "$(CYAN)📦 Creando entorno virtual...$(RESET)"; \
		$(PYTHON) -m venv venv; \
		echo "$(CYAN)📦 Instalando dependencias en venv...$(RESET)"; \
		venv/bin/pip install --quiet 'numpy<2' matplotlib pandas; \
		echo "$(GREEN)✅ Entorno virtual creado$(RESET)"; \
	fi

.PHONY: visualize
visualize: venv
	@echo "$(CYAN)📊 Generando curvas de aprendizaje...$(RESET)"
	@if [ ! -f "training_history.json" ]; then \
		echo "$(RED)❌ Error: training_history.json no encontrado$(RESET)"; \
		echo "$(YELLOW)   Ejecuta 'make train' primero$(RESET)"; \
		exit 1; \
	fi
	@venv/bin/python visualize.py --history training_history.json
	@echo "$(GREEN)✅ Curvas guardadas: loss_curve.png, accuracy_curve.png$(RESET)"

# ============================================================================
# Testing
# ============================================================================

.PHONY: test
test: split
	@echo "$(CYAN)🧪 Ejecutando test de integración...$(RESET)"
	@$(PYTHON) test_integration.py

.PHONY: test-train
test-train:
	@echo "$(CYAN)🧪 Test rápido de entrenamiento (5 épocas)...$(RESET)"
	@$(PYTHON) train.py --dataset $(TRAIN_DATA) \
		--layers 30 16 16 2 \
		--epochs 5 \
		--learning_rate 0.05 \
		--batch_size 16 \
		--activation relu

# ============================================================================
# Limpieza
# ============================================================================

.PHONY: clean
clean:
	@echo "$(YELLOW)🧹 Limpiando archivos generados...$(RESET)"
	@rm -f $(TRAIN_DATA) $(VALID_DATA)
	@rm -f training_history.json metrics.json
	@rm -f *.png
	@rm -rf __pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Limpieza completada$(RESET)"

.PHONY: fclean
fclean: clean
	@echo "$(RED)🗑️  Limpieza completa (incluye modelo y venv)...$(RESET)"
	@rm -f $(MODEL)
	@rm -rf venv
	@echo "$(GREEN)✅ Limpieza completa finalizada$(RESET)"
	@rm -f *.npy
	@echo "$(GREEN)✅ Limpieza completa terminada$(RESET)"

.PHONY: re
re: fclean all

# ============================================================================
# Desarrollo
# ============================================================================

.PHONY: check
check:
	@echo "$(CYAN)🔍 Verificando código...$(RESET)"
	@$(PYTHON) -m py_compile *.py 2>/dev/null && \
	echo "$(GREEN)✅ Sintaxis correcta en todos los archivos$(RESET)" || \
	echo "$(RED)❌ Errores de sintaxis detectados$(RESET)"

.PHONY: help
help: info

# ============================================================================
# Información
# ============================================================================

.PHONY: status
status:
	@echo "$(CYAN)📊 Estado del proyecto:$(RESET)"
	@echo ""
	@echo "  Dataset original:  $(DATASET) $$(if [ -f $(DATASET) ]; then echo '$(GREEN)✓$(RESET)'; else echo '$(RED)✗$(RESET)'; fi)"
	@echo "  Training data:     $(TRAIN_DATA) $$(if [ -f $(TRAIN_DATA) ]; then echo '$(GREEN)✓$(RESET)'; else echo '$(RED)✗$(RESET)'; fi)"
	@echo "  Validation data:   $(VALID_DATA) $$(if [ -f $(VALID_DATA) ]; then echo '$(GREEN)✓$(RESET)'; else echo '$(RED)✗$(RESET)'; fi)"
	@echo "  Modelo entrenado:  $(MODEL) $$(if [ -f $(MODEL) ]; then echo '$(GREEN)✓$(RESET)'; else echo '$(RED)✗$(RESET)'; fi)"
	@echo ""
