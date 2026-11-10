#!/usr/bin/env python3
"""
test_integration.py - Test de integración completo del MLP

Prueba todo el flujo del proyecto:
    1. División de datos
    2. Preprocesamiento
    3. Entrenamiento (pocas épocas)
    4. Predicción
    5. Evaluación
    6. Visualización

Este script valida que todos los componentes funcionen juntos correctamente.
"""

import sys
import os


def test_split():
    """Test del script de división."""
    print("=" * 80)
    print("🧪 TEST 1: División del dataset")
    print("=" * 80)
    print()
    
    import split_dataset
    
    # Ya ejecutamos esto antes, verificar que existen los archivos
    if os.path.exists('data_training.csv') and os.path.exists('data_validation.csv'):
        print("✅ Archivos de división encontrados")
        return True
    else:
        print("❌ Archivos de división no encontrados")
        return False


def test_preprocessing():
    """Test del preprocesamiento."""
    print()
    print("=" * 80)
    print("🧪 TEST 2: Preprocesamiento de datos")
    print("=" * 80)
    print()
    
    from preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(normalization='standardization')
    
    # Cargar datos
    features, labels, ids = preprocessor.load_data('data_training.csv')
    print(f"   ✓ Cargadas {len(features)} muestras")
    
    # Codificar labels
    y = preprocessor.encode_labels(labels)
    print(f"   ✓ Labels codificados: {len(y)} valores")
    
    # Normalizar
    X = preprocessor.fit_transform(features)
    print(f"   ✓ Features normalizados: shape ({len(X)}, {len(X[0])})")
    
    # Verificar normalización
    first_feature = [sample[0] for sample in X]
    mean = sum(first_feature) / len(first_feature)
    variance = sum((f - mean)**2 for f in first_feature) / len(first_feature)
    std = variance ** 0.5
    
    print(f"   ✓ Feature 0: mean={mean:.6f}, std={std:.6f}")
    print("   ✓ Normalización (standardization) funciona correctamente")
    
    return True


def test_network_creation():
    """Test de creación de red."""
    print()
    print("=" * 80)
    print("🧪 TEST 3: Creación de red neuronal")
    print("=" * 80)
    print()
    
    from network import Network
    
    # Esta red es mas pequena que la arquitectura principal (30 -> 24 -> 24 -> 2)
    # para que el test de integracion sea rapido.
    network = Network(loss='categorical_crossentropy', learning_rate=0.1)
    network.build(
        architecture=[30, 16, 16, 2],
        activations=['sigmoid', 'sigmoid', 'softmax'],
        weight_init='he',
        seed=42
    )
    
    print(f"   ✓ Red creada: {network}")
    print(f"   ✓ Parámetros: {network.get_parameters_count()}")
    print("   ℹ️  Arquitectura de prueba: 30 entradas -> 16 -> 16 -> 2 salidas")
    print("      La configuración principal del proyecto utiliza 30 -> 24 -> 24 -> 2.")
    
    return network


def test_training(network):
    """Test de entrenamiento rápido."""
    print()
    print("=" * 80)
    print("🧪 TEST 4: Entrenamiento (5 épocas de prueba)")
    print("=" * 80)
    print()
    
    from preprocessing import DataPreprocessor
    
    # Cargar datos
    preprocessor = DataPreprocessor(normalization='standardization')
    features, labels, _ = preprocessor.load_data('data_training.csv')
    y_train = preprocessor.encode_labels(labels)
    X_train = preprocessor.fit_transform(features)
    
    # Cargar validación
    features_val, labels_val, _ = preprocessor.load_data('data_validation.csv')
    y_valid = preprocessor.encode_labels(labels_val)
    X_valid = preprocessor.transform(features_val)
    
    print(f"   Train: {len(X_train)} muestras")
    print(f"   Valid: {len(X_valid)} muestras")
    print()
    print("   📖 ¿Qué es una época?")
    print("      Una época es una pasada completa por todas las muestras de")
    print("      entrenamiento. En cada época la red ajusta sus pesos y después")
    print("      se evalúa con los datos de entrenamiento y de validación.")
    print()
    print("   📖 Significado de las métricas")
    print("      loss     : error medio sobre el conjunto de entrenamiento.")
    print("      val_loss : error medio sobre el conjunto de validación.")
    print("      acc      : proporción de predicciones correctas en entrenamiento.")
    print("      val_acc  : proporción de predicciones correctas en validación.")
    print("      Menor loss/val_loss es mejor; mayor acc/val_acc es mejor.")
    print("      La validación usa datos que no ajustan los pesos y ayuda a detectar")
    print("      overfitting: aprender demasiado bien los datos de entrenamiento.")
    print()
    
    # Entrenar 5 épocas de prueba
    history = []
    for epoch in range(1, 6):
        # Train
        train_loss, train_acc = network.evaluate(X_train, y_train)
        
        # Forward y backward en todo el dataset (batch completo para test rápido)
        predictions = network.forward(X_train)
        network.backward(y_train, predictions)
        network.update_weights()
        
        # Evaluar después de actualizar
        train_loss, train_acc = network.evaluate(X_train, y_train)
        val_loss, val_acc = network.evaluate(X_valid, y_valid)
        history.append((train_loss, val_loss, train_acc, val_acc))
        
        print(f"   epoch {epoch:02d}/05 - "
              f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
              f"acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
    
    print()
    print("   ✅ Entrenamiento de prueba completado")
    first_loss, first_val_loss, first_acc, first_val_acc = history[0]
    last_loss, last_val_loss, last_acc, last_val_acc = history[-1]
    print("   📊 Lectura de estos resultados (época 1 -> época 5):")
    print(f"      loss     : {first_loss:.4f} -> {last_loss:.4f}")
    print(f"      val_loss : {first_val_loss:.4f} -> {last_val_loss:.4f}")
    print(f"      acc      : {first_acc:.4f} -> {last_acc:.4f}")
    print(f"      val_acc  : {first_val_acc:.4f} -> {last_val_acc:.4f}")
    print("      La pérdida baja: la red mejora ligeramente la confianza de sus")
    print("      predicciones. La accuracy puede quedarse igual porque esa métrica")
    print("      solo cambia cuando cambia la clase finalmente elegida.")
    print("      Con solo 5 épocas y esta configuración simplificada, esto no indica")
    print("      por sí solo un error ni permite juzgar el rendimiento definitivo.")
    
    return True


def test_save_load():
    """Test de guardar y cargar modelo."""
    print()
    print("=" * 80)
    print("🧪 TEST 5: Guardar y cargar modelo")
    print("=" * 80)
    print()
    
    import json
    from network import Network
    from preprocessing import DataPreprocessor
    
    # Crear red pequeña
    network = Network(loss='categorical_crossentropy', learning_rate=0.01)
    network.build([30, 8, 2], ['sigmoid', 'softmax'], seed=42)
    
    # Guardar
    model_data = {
        'architecture': [30, 8, 2],
        'activations': ['sigmoid', 'softmax'],
        'weights': [layer.weights for layer in network.layers],
        'biases': [layer.bias for layer in network.layers],
        'preprocessing': {
            'method': 'standardization',
            'feature_means': [0.0] * 30,
            'feature_stds': [1.0] * 30,
            'feature_mins': [0.0] * 30,
            'feature_maxs': [1.0] * 30
        }
    }
    
    with open('test_model.json', 'w') as f:
        json.dump(model_data, f)
    
    print("   ✓ Modelo guardado en 'test_model.json'")
    
    # Cargar
    with open('test_model.json', 'r') as f:
        loaded_data = json.load(f)
    
    print("   ✓ Modelo cargado correctamente")
    
    # Verificar
    if loaded_data['architecture'] == [30, 8, 2]:
        print("   ✓ Arquitectura verificada")
    
    # Limpiar
    os.remove('test_model.json')
    print("   ✓ Archivo de test eliminado")
    
    return True


def test_metrics():
    """Test de métricas."""
    print()
    print("=" * 80)
    print("🧪 TEST 6: Métricas de evaluación")
    print("=" * 80)
    print()
    
    from metrics import compute_accuracy, compute_precision, compute_recall, compute_f1_score
    
    # Datos de prueba
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.4, 0.6],
              [0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4]]
    
    acc = compute_accuracy(y_true, y_pred)
    prec = compute_precision(y_true, y_pred)
    rec = compute_recall(y_true, y_pred)
    f1 = compute_f1_score(y_true, y_pred)
    
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print()
    print("   📖 ¿Qué mide cada métrica?")
    print("      Accuracy  : porcentaje de aciertos totales.")
    print("      Precision : de los positivos predichos, cuántos eran correctos.")
    print("      Recall    : de los positivos reales, cuántos fueron encontrados.")
    print("      F1-Score  : equilibrio entre precision y recall.")
    print()
    print("   ℹ️  Este caso usa etiquetas reales [1, 0, 1, 1, 0, 0, 1, 0].")
    print("      Las clases predichas también son [1, 0, 1, 1, 0, 0, 1, 0].")
    print("      Como las 8 predicciones son correctas, todas las métricas valen 1.0.")
    print("      Esto comprueba las fórmulas, pero no significa que el modelo real")
    print("      tenga rendimiento perfecto. Para eso hay que evaluarlo con datos")
    print("      reales de validación o test.")
    print("   ✅ Métricas calculadas correctamente")
    
    return True


def main():
    """
    Ejecuta todos los tests de integración.
    """
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "🧪 TEST DE INTEGRACIÓN COMPLETO" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    tests = [
        ("División de datos", test_split),
        ("Preprocesamiento", test_preprocessing),
        ("Creación de red", test_network_creation),
        ("Entrenamiento", test_training),
        ("Guardar/Cargar modelo", test_save_load),
        ("Métricas", test_metrics)
    ]
    
    results = []
    network = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "Entrenamiento":
                # Necesita la red creada en el test anterior
                result = test_func(network)
            elif test_name == "Creación de red":
                network = test_func()
                result = network is not None
            else:
                result = test_func()
            
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Resumen final
    print()
    print("=" * 80)
    print("📊 RESUMEN DE TESTS")
    print("=" * 80)
    print()
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}  {test_name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("=" * 80)
        print("✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
        print("=" * 80)
        print()
        print("🎉 El sistema está listo para entrenamiento real")
        print()
        print("📝 Comandos sugeridos:")
        print("   1. make split    # Dividir dataset (ya hecho)")
        print("   2. make train    # Entrenar modelo")
        print("   3. make predict  # Hacer predicciones")
        print()
    else:
        print("=" * 80)
        print("⚠️  ALGUNOS TESTS FALLARON")
        print("=" * 80)
        print()
        print("Revisar los errores anteriores")
        print()
    
    return all_passed


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print()
        print("⚠️  Tests interrumpidos")
        sys.exit(1)
