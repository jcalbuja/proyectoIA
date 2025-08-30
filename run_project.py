import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import os
import numpy as np
import matplotlib.pyplot as plt

# Importar las funciones de los módulos que creamos en `src/`
from src.utils import preprocess_data
from src.model_training import train_decision_tree, train_neural_network

if __name__ == "__main__":
    print("Iniciando el pipeline de análisis de ciberseguridad...")
    
    # 1. Cargar el dataset
    file_path = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo en {file_path}. Asegúrate de que el dataset está en la carpeta 'data/'.")
    else:
        df = pd.read_csv(file_path)
        print("Dataset cargado exitosamente.")
        
        # 2. Preprocesamiento de datos
        X, y, label_encoder = preprocess_data(df)
        print("Preprocesamiento de datos completado.")
        
        # Separar los datos por clase para una división manual y segura
        X_ddos = X[y == 1]
        y_ddos = y[y == 1]

        # VERIFICACIÓN CRÍTICA: ¿Hay muestras de la clase 'DDoS'?
        if len(X_ddos) == 0:
            print("\n")
            print("=======================================================================")
            print("¡ERROR DE DATOS! No se encontraron muestras de la clase 'DDoS' en el dataset.")
            print("No es posible entrenar o evaluar un modelo para detectar ataques.")
            print("Por favor, revisa que el archivo CSV contenga ambas clases (BENIGN y DDoS).")
            print("El script se detendrá ahora.")
            print("=======================================================================")
        else:
            # Continuar con el pipeline solo si hay datos para ambas clases
            X_benign = X[y == 0]
            y_benign = y[y == 0]
            
            # Tomar 25 muestras de ambas clases para el conjunto de prueba
            test_size_per_class = min(25, len(X_ddos))
            
            X_test_ddos, X_train_ddos, y_test_ddos, y_train_ddos = train_test_split(
                X_ddos, y_ddos, test_size=test_size_per_class, random_state=42
            )

            X_test_benign, X_train_benign, y_test_benign, y_train_benign = train_test_split(
                X_benign, y_benign, test_size=test_size_per_class, random_state=42
            )
            
            # Unir los conjuntos de entrenamiento y prueba usando np.concatenate
            X_train = np.concatenate([X_train_benign, X_train_ddos], axis=0)
            y_train = np.concatenate([y_train_benign, y_train_ddos], axis=0)
            X_test = np.concatenate([X_test_benign, X_test_ddos], axis=0)
            y_test = np.concatenate([y_test_benign, y_test_ddos], axis=0)
            
            print("Datos divididos en conjuntos de entrenamiento y prueba de forma manual y segura.")
            
            # 3. Entrenamiento del Modelo Superficial (Árbol de Decisión)
            print("\nEntrenando Árbol de Decisión...")
            best_clf = train_decision_tree(X_train, y_train)
            y_pred_clf = best_clf.predict(X_test)
            print("Árbol de Decisión entrenado y evaluado.")
            
            # 4. Entrenamiento del Modelo Profundo (Red Neuronal)
            print("\nEntrenando Red Neuronal...")
            model_nn, X_test_scaled = train_neural_network(X_train, y_train, X_test)
            y_pred_nn_prob = model_nn.predict(X_test_scaled, verbose=0)
            y_pred_nn = (y_pred_nn_prob > 0.5).astype("int32")
            print("Red Neuracional entrenada y evaluada.")
            
            # 5. Evaluación y comparación de modelos
            print("\n--- Resultados del Árbol de Decisión ---")
            print(classification_report(y_test, y_pred_clf, target_names=label_encoder.classes_))
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_clf, labels=np.unique(y)))
            
            print("\n--- Resultados de la Red Neuronal ---")
            print(classification_report(y_test, y_pred_nn, target_names=label_encoder.classes_))
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_nn, labels=np.unique(y)))
            
            # 6. Generar la Curva ROC y el AUC
            # Obtener las probabilidades de predicción para la clase positiva (DDoS)
            y_pred_clf_prob = best_clf.predict_proba(X_test)[:, 1]
            
            # La Red Neuronal ya nos da las probabilidades en `y_pred_nn_prob`
            
            # Calcular la Curva ROC y el AUC
            fpr_clf, tpr_clf, _ = roc_curve(y_test, y_pred_clf_prob)
            auc_clf = roc_auc_score(y_test, y_pred_clf_prob)

            fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_nn_prob)
            auc_nn = roc_auc_score(y_test, y_pred_nn_prob)
            
            print("\n--- Curva ROC y AUC ---")
            print(f"AUC del Árbol de Decisión: {auc_clf:.4f}")
            print(f"AUC de la Red Neuronal: {auc_nn:.4f}")

            # Graficar la Curva ROC
            plt.figure(figsize=(10, 8))
            plt.plot(fpr_clf, tpr_clf, label=f'Árbol de Decisión (AUC = {auc_clf:.2f})', color='blue')
            plt.plot(fpr_nn, tpr_nn, label=f'Red Neuronal (AUC = {auc_nn:.2f})', color='red')
            plt.plot([0, 1], [0, 1], 'k--', label='Clasificador aleatorio (AUC = 0.50)', color='gray')
            
            plt.xlabel('Tasa de Falsos Positivos (FPR)')
            plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
            plt.title('Curva ROC de Detección de Ataques DDoS')
            plt.legend(loc='lower right')
            plt.grid(True)
          #  plt.show()
            plt.savefig('roc_curve.png')

            # Métricas finales
            accuracy_clf = accuracy_score(y_test, y_pred_clf)
            accuracy_nn = accuracy_score(y_test, y_pred_nn)
            print(f"\nPrecisión del Árbol de Decisión: {accuracy_clf:.4f}")
            print(f"Precisión de la Red Neuronal: {accuracy_nn:.4f}")