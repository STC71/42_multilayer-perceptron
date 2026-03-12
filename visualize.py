#!/usr/bin/env python3
"""
visualize.py - Visualización de curvas de aprendizaje

REQUERIDO POR EL SUBJECT:
    - Curva de Loss (training y validation)
    - Curva de Accuracy (training y validation)
    
Genera dos gráficos según el subject (página 9, Figuras IV.1 y IV.2)

Usage:
    python3 visualize.py --history training_history.json
    python3 visualize.py --history training_history.json --show
"""

import sys
import argparse
import json

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib no disponible. Instalar con: pip install matplotlib", file=sys.stderr)


def load_history(filepath):
    """
    Carga el historial de entrenamiento.
    
    Args:
        filepath: Ruta al archivo JSON con historial
        
    Returns:
        Dict con historial
    """
    try:
        with open(filepath, 'r') as f:
            history = json.load(f)
        return history
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error al parsear historial: {e}", file=sys.stderr)
        sys.exit(1)


def plot_loss_curve(history, save_path='loss_curve.png', show=False):
    """
    Genera la curva de Loss (según subject, Figura IV.1).
    
    Muestra:
        - Training Loss (azul)
        - Validation Loss (naranja)
    
    Args:
        history: Dict con historial de entrenamiento
        save_path: Ruta donde guardar el gráfico
        show: Si True, muestra el gráfico interactivo
    """
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib no disponible, no se puede generar gráfico")
        return
    
    epochs = range(1, len(history['loss']) + 1)
    train_loss = history['loss']
    val_loss = history.get('val_loss', [])
    
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    
    # Plot validation loss si está disponible
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    
    # Configuración del gráfico (según subject)
    plt.title('Model Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Guardar
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Curva de Loss guardada en '{save_path}'")
    
    if show:
        plt.show()
    
    plt.close()


def plot_accuracy_curve(history, save_path='accuracy_curve.png', show=False):
    """
    Genera la curva de Accuracy (según subject, Figura IV.2).
    
    Muestra:
        - Training Accuracy (azul)
        - Validation Accuracy (naranja)
    
    Args:
        history: Dict con historial de entrenamiento
        save_path: Ruta donde guardar el gráfico
        show: Si True, muestra el gráfico interactivo
    """
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib no disponible, no se puede generar gráfico")
        return
    
    epochs = range(1, len(history['accuracy']) + 1)
    train_acc = history['accuracy']
    val_acc = history.get('val_accuracy', [])
    
    plt.figure(figsize=(10, 6))
    
    # Plot training accuracy
    plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
    
    # Plot validation accuracy si está disponible
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
    
    # Configuración del gráfico (según subject)
    plt.title('Model Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Añadir línea de referencia en 1.0 (perfección)
    plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, linewidth=1)
    
    # Guardar
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Curva de Accuracy guardada en '{save_path}'")
    
    if show:
        plt.show()
    
    plt.close()


def plot_combined_curves(history, save_path='learning_curves.png', show=False):
    """
    Genera un gráfico combinado con Loss y Accuracy en subplots.
    
    BONUS: Visualización más completa en una sola imagen.
    
    Args:
        history: Dict con historial
        save_path: Ruta donde guardar
        show: Si mostrar interactivamente
    """
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib no disponible, no se puede generar gráfico")
        return
    
    epochs = range(1, len(history['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Loss
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    ax2.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    if 'val_accuracy' in history and history['val_accuracy']:
        ax2.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, linewidth=1)
    
    # Guardar
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Curvas combinadas guardadas en '{save_path}'")
    
    if show:
        plt.show()
    
    plt.close()


def print_training_summary(history):
    """
    Imprime un resumen del entrenamiento.
    
    Args:
        history: Dict con historial
    """
    print("=" * 80)
    print("📊 RESUMEN DEL ENTRENAMIENTO")
    print("=" * 80)
    print()
    
    n_epochs = len(history['loss'])
    
    print(f"📈 Épocas entrenadas: {n_epochs}")
    print()
    
    # Mejor época
    if 'val_loss' in history and history['val_loss']:
        best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        best_val_loss = history['val_loss'][best_epoch - 1]
        best_val_acc = history['val_accuracy'][best_epoch - 1]
        
        print(f"🏆 Mejor época: {best_epoch}")
        print(f"   Validation Loss:     {best_val_loss:.6f}")
        print(f"   Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Última época
    print()
    print(f"📍 Última época: {n_epochs}")
    print(f"   Training Loss:       {history['loss'][-1]:.6f}")
    print(f"   Training Accuracy:   {history['accuracy'][-1]:.4f} ({history['accuracy'][-1]*100:.2f}%)")
    
    if 'val_loss' in history and history['val_loss']:
        print(f"   Validation Loss:     {history['val_loss'][-1]:.6f}")
        print(f"   Validation Accuracy: {history['val_accuracy'][-1]:.4f} ({history['val_accuracy'][-1]*100:.2f}%)")
    
    # Mejora
    print()
    print(f"📉 Mejora:")
    initial_loss = history['loss'][0]
    final_loss = history['loss'][-1]
    improvement = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"   Loss inicial:  {initial_loss:.6f}")
    print(f"   Loss final:    {final_loss:.6f}")
    print(f"   Mejora:        {improvement:.2f}%")
    
    print()
    print("=" * 80)
    print()


def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Visualiza curvas de aprendizaje del entrenamiento',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--history',
        type=str,
        default='training_history.json',
        help='Archivo JSON con historial de entrenamiento'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar gráficos interactivos (además de guardarlos)'
    )
    
    parser.add_argument(
        '--loss_output',
        type=str,
        default='loss_curve.png',
        help='Archivo de salida para curva de loss'
    )
    
    parser.add_argument(
        '--accuracy_output',
        type=str,
        default='accuracy_curve.png',
        help='Archivo de salida para curva de accuracy'
    )
    
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Generar también gráfico combinado (bonus)'
    )
    
    return parser.parse_args()


def main():
    """
    Función principal de visualización.
    """
    # Parsear argumentos
    args = parse_arguments()
    
    if not HAS_MATPLOTLIB:
        print("❌ Error: matplotlib no está instalado")
        print("   Instalar con: pip install matplotlib")
        sys.exit(1)
    
    print("=" * 80)
    print("📊 VISUALIZACIÓN DE CURVAS DE APRENDIZAJE")
    print("=" * 80)
    print()
    
    # Cargar historial
    print(f"📂 Cargando historial desde '{args.history}'...")
    history = load_history(args.history)
    print(f"   ✓ Historial cargado correctamente")
    print()
    
    # Mostrar resumen
    print_training_summary(history)
    
    # Generar gráficos (REQUERIDO POR EL SUBJECT)
    print("🎨 Generando curvas de aprendizaje...")
    print()
    
    # Curva de Loss (Figura IV.1 del subject)
    plot_loss_curve(history, save_path=args.loss_output, show=False)
    
    # Curva de Accuracy (Figura IV.2 del subject)
    plot_accuracy_curve(history, save_path=args.accuracy_output, show=False)
    
    # Curvas combinadas (BONUS)
    if args.combined:
        plot_combined_curves(history, save_path='learning_curves.png', show=False)
    
    print()
    
    # Mostrar gráficos si se solicita
    if args.show:
        print("🖼️  Mostrando gráficos...")
        if HAS_MATPLOTLIB:
            # Volver a generar con show=True
            plot_loss_curve(history, save_path=args.loss_output, show=True)
            plot_accuracy_curve(history, save_path=args.accuracy_output, show=True)
    
    print("=" * 80)
    print("✅ VISUALIZACIÓN COMPLETADA")
    print("=" * 80)
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("⚠️  Visualización interrumpida")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error durante visualización: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
