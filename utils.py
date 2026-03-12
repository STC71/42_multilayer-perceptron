#!/usr/bin/env python3
"""
utils.py - Utilidades generales para el proyecto Multilayer Perceptron

Incluye:
    - Funciones matemáticas (no incluidas en numpy)
    - Manejo de archivos
    - Formateo de output
    - Helpers generales
"""

import sys
import time


def ensure_list_format(data):
    """
    Asegura que los datos estén en formato lista de listas.
    
    Args:
        data: Datos de entrada (puede ser lista, lista de listas, etc.)
        
    Returns:
        Lista de listas
    """
    if not isinstance(data, list):
        return [[data]]
    
    if len(data) == 0:
        return [[]]
    
    if not isinstance(data[0], list):
        return [data]
    
    return data


def save_model(filepath, weights, biases, config):
    """
    Guarda el modelo (pesos, biases y configuración) en formato nativo.
    
    Args:
        filepath: Ruta donde guardar el modelo
        weights: Lista de matrices de pesos
        biases: Lista de vectores de bias
        config: Dict con configuración del modelo
    """
    try:
        # Formato: línea por línea en texto plano
        with open(filepath, 'w') as f:
            # Guardar configuración
            f.write("CONFIG\n")
            for key, value in config.items():
                f.write(f"{key}:{value}\n")
            f.write("END_CONFIG\n")
            
            # Guardar pesos y biases por capa
            for layer_idx, (W, b) in enumerate(zip(weights, biases)):
                f.write(f"LAYER_{layer_idx}\n")
                
                # Guardar dimensiones de W
                rows = len(W)
                cols = len(W[0]) if rows > 0 else 0
                f.write(f"WEIGHTS:{rows},{cols}\n")
                
                # Guardar matriz de pesos
                for row in W:
                    f.write(','.join(str(val) for val in row) + '\n')
                
                # Guardar bias
                f.write(f"BIAS:{len(b)}\n")
                f.write(','.join(str(val) for val in b) + '\n')
                
                f.write(f"END_LAYER_{layer_idx}\n")
        
        print(f"✅ Modelo guardado en '{filepath}'")
        return True
        
    except Exception as e:
        print(f"❌ Error al guardar modelo: {e}", file=sys.stderr)
        return False


def load_model(filepath):
    """
    Carga el modelo desde un archivo.
    
    Args:
        filepath: Ruta del archivo del modelo
        
    Returns:
        Tupla (weights, biases, config)
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        weights = []
        biases = []
        config = {}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "CONFIG":
                i += 1
                while i < len(lines) and lines[i].strip() != "END_CONFIG":
                    key_val = lines[i].strip().split(':', 1)
                    if len(key_val) == 2:
                        key, val = key_val
                        # Intentar convertir a número si es posible
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                        config[key] = val
                    i += 1
            
            elif line.startswith("LAYER_"):
                i += 1
                
                # Leer dimensiones de pesos
                if lines[i].startswith("WEIGHTS:"):
                    dims = lines[i].strip().split(':')[1].split(',')
                    rows, cols = int(dims[0]), int(dims[1])
                    i += 1
                    
                    # Leer matriz de pesos
                    W = []
                    for _ in range(rows):
                        row = [float(val) for val in lines[i].strip().split(',')]
                        W.append(row)
                        i += 1
                    weights.append(W)
                
                # Leer bias
                if lines[i].startswith("BIAS:"):
                    size = int(lines[i].strip().split(':')[1])
                    i += 1
                    b = [float(val) for val in lines[i].strip().split(',')]
                    biases.append(b)
                    i += 1
            
            i += 1
        
        print(f"✅ Modelo cargado desde '{filepath}'")
        return weights, biases, config
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}", file=sys.stderr)
        sys.exit(1)


def format_time(seconds):
    """
    Formatea tiempo en segundos a formato legible.
    
    Args:
        seconds: Tiempo en segundos
        
    Returns:
        String formateado (ej: "2h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Muestra una barra de progreso en la terminal.
    
    Args:
        iteration: Iteración actual
        total: Total de iteraciones
        prefix: String prefijo
        suffix: String sufijo
        length: Longitud de la barra
        fill: Carácter de relleno
    """
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    
    # Imprimir nueva línea al completar
    if iteration == total:
        print()


class Timer:
    """Clase simple para medir tiempos de ejecución."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Inicia el cronómetro."""
        self.start_time = time.time()
    
    def stop(self):
        """Detiene el cronómetro."""
        self.end_time = time.time()
    
    def elapsed(self):
        """Retorna el tiempo transcurrido en segundos."""
        if self.start_time is None:
            return 0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def elapsed_formatted(self):
        """Retorna el tiempo transcurrido formateado."""
        return format_time(self.elapsed())


def validate_file_exists(filepath):
    """
    Valida que un archivo exista.
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        True si existe, False en caso contrario
    """
    import os
    return os.path.isfile(filepath)


def create_directory_if_not_exists(dirpath):
    """
    Crea un directorio si no existe.
    
    Args:
        dirpath: Ruta del directorio
    """
    import os
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"✅ Directorio creado: {dirpath}")


def print_section_header(title, char='=', width=80):
    """
    Imprime un header de sección formateado.
    
    Args:
        title: Título de la sección
        char: Carácter para la línea
        width: Ancho total
    """
    print()
    print(char * width)
    print(title.center(width))
    print(char * width)
    print()


def print_metric(name, value, width=20):
    """
    Imprime una métrica formateada.
    
    Args:
        name: Nombre de la métrica
        value: Valor de la métrica
        width: Ancho para el nombre
    """
    if isinstance(value, float):
        print(f"   {name:<{width}} {value:8.6f}")
    else:
        print(f"   {name:<{width}} {value}")
