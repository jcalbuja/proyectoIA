from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers
from keras import Sequential

def train_decision_tree(X_train, y_train):
    """
    Entrena un Árbol de Decisión con optimización de hiperparámetros.
    Utiliza 'f1_weighted' para manejar el desbalance de clases.
    
    Args:
        X_train (pd.DataFrame): Datos de entrenamiento (características).
        y_train (pd.Series): Etiquetas de entrenamiento.
        
    Returns:
        sklearn.tree.DecisionTreeClassifier: El modelo de Árbol de Decisión con los mejores hiperparámetros.
    """
    print("Iniciando la búsqueda de hiperparámetros para el Árbol de Decisión...")
    clf = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
    }
    # Cambio aquí: se usa 'f1_weighted' en lugar de 'f1'
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_neural_network(X_train, y_train, X_test):
    """
    Entrena una Red Neuronal Densa con Keras.
    
    Args:
        X_train (pd.DataFrame): Datos de entrenamiento (características).
        y_train (pd.Series): Etiquetas de entrenamiento.
        X_test (pd.DataFrame): Datos de prueba para el escalado.

    Returns:
        tuple: El modelo de Keras entrenado y los datos de prueba escalados.
    """
    print("Iniciando el entrenamiento de la Red Neuronal...")
    # Escalado de datos para la Red Neuronal
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Construcción del modelo de red neuronal
    model = Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid') # Capa de salida para clasificación binaria
    ])
    
    # Compilación y entrenamiento
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    return model, X_test_scaled