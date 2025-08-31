import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import os
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

# Importamos las funciones de los módulos que creamos en `src/`
from src.utils import preprocess_data
from src.model_training import train_decision_tree, train_neural_network

if __name__ == "__main__":
    print("Iniciamos el pipeline de análisis de ciberseguridad...")
    
    # 1. Cargamos el dataset
    file_path = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo en {file_path}. Asegúrate de que el dataset está en la carpeta 'data/'.")
    else:
        df = pd.read_csv(file_path)
        print("Dataset cargado exitosamente.")
        
        # 2. Preprocesamiento de datos
        X, y, label_encoder = preprocess_data(df)
        print("Preprocesamiento de datos completado.")

        # Dividimos los datos en conjuntos de entrenamiento y prueba de manera estratificada
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # VERIFICACIÓN CRÍTICA: ¿Hay muestras de la clase 'DDoS' en el conjunto de prueba?
        if len(np.unique(y_test)) < 2:
            print("\n")
            print("=======================================================================")
            print("¡ERROR DE DATOS! El conjunto de prueba no contiene ambas clases.")
            print("No es posible entrenar o evaluar un modelo para detectar ataques.")
            print("Por favor, revisa que el archivo CSV contenga ambas clases (BENIGN y DDoS).")
            print("El script se detendrá ahora.")
            print("=======================================================================")
        else:
            # Balanceamos el conjunto de entrenamiento usando RandomOverSampler
            print("Balanceando el conjunto de entrenamiento con RandomOverSampler...")
            oversampler = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

            print(f"Tamaño original del conjunto de entrenamiento: {len(X_train)}")
            print(f"Tamaño balanceado del conjunto de entrenamiento: {len(X_train_resampled)}")
            print("Datos divididos y balanceados exitosamente.")
            
            # 3. Entrenamiento del Modelo Superficial (Árbol de Decisión)
            print("\nEntrenando Árbol de Decisión...")
            best_clf = train_decision_tree(X_train_resampled, y_train_resampled)
            y_pred_clf = best_clf.predict(X_test)
            print("Árbol de Decisión entrenado y evaluado.")
            
            # 4. Entrenamiento del Modelo Profundo (Red Neuronal)
            print("\nEntrenando Red Neuronal...")
            model_nn, X_test_scaled, history_nn = train_neural_network(X_train_resampled, y_train_resampled, X_test)
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
            
            # 6. Generamos la Curva ROC y el AUC
            # Obtenemos las probabilidades de predicción para la clase positiva (DDoS)
            y_pred_clf_prob = best_clf.predict_proba(X_test)[:, 1]
            y_pred_nn_prob = model_nn.predict(X_test_scaled, verbose=0)
            
            # Calculamos la Curva ROC y el AUC
            fpr_clf, tpr_clf, _ = roc_curve(y_test, y_pred_clf_prob)
            auc_clf = roc_auc_score(y_test, y_pred_clf_prob)

            fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_nn_prob)
            auc_nn = roc_auc_score(y_test, y_pred_nn_prob)
            
            print("\n--- Curva ROC y AUC ---")
            print(f"AUC del Árbol de Decisión: {auc_clf:.4f}")
            print(f"AUC de la Red Neuronal: {auc_nn:.4f}")

            # Graficamos la Curva ROC y guardarla
            plt.figure(figsize=(10, 8))
            plt.plot(fpr_clf, tpr_clf, label=f'Árbol de Decisión (AUC = {auc_clf:.2f})', color='blue')
            plt.plot(fpr_nn, tpr_nn, label=f'Red Neuronal (AUC = {auc_nn:.2f})', color='red')
            plt.plot([0, 1], [0, 1], 'k--', label='Clasificador aleatorio (AUC = 0.50)', color='gray')
            
            plt.xlabel('Tasa de Falsos Positivos (FPR)')
            plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
            plt.title('Curva ROC de Detección de Ataques DDoS')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.savefig('roc_curve.png')

            # 7. Graficamos las curvas de aprendizaje de la Red Neuronal
            print("\n--- Curvas de Aprendizaje de la Red Neuronal ---")
            plt.figure(figsize=(12, 6))

            # Gráfica Curva de Precisión
            plt.subplot(1, 2, 1)
            plt.plot(history_nn['accuracy'], label='Precisión de Entrenamiento')
            plt.title('Precisión del Modelo')
            plt.ylabel('Precisión')
            plt.xlabel('Época')
            plt.legend()

            # Gráfica Curva de Pérdida (Loss)
            plt.subplot(1, 2, 2)
            plt.plot(history_nn['loss'], label='Pérdida de Entrenamiento')
            plt.title('Pérdida del Modelo')
            plt.ylabel('Pérdida')
            plt.xlabel('Época')
            plt.legend()

            plt.tight_layout()
            plt.savefig('learning_curve.png')
            
            # Métricas finales entre ambos modelos
            accuracy_clf = accuracy_score(y_test, y_pred_clf)
            accuracy_nn = accuracy_score(y_test, y_pred_nn)
            print(f"\nPrecisión del Árbol de Decisión: {accuracy_clf:.4f}")
            print(f"Precisión de la Red Neuronal: {accuracy_nn:.4f}")