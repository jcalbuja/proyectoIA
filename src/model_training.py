from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

def train_decision_tree(X_train, y_train):
    """
    Entrena un modelo de Árbol de Decisión con búsqueda aleatoria de hiperparámetros.
    
    Args:
        X_train (np.array): Conjunto de entrenamiento de características.
        y_train (np.array): Conjunto de entrenamiento de etiquetas.
        
    Returns:
        DecisionTreeClassifier: El mejor modelo de Árbol de Decisión entrenado.
    """
    print("Iniciando la búsqueda aleatoria de hiperparámetros para el Árbol de Decisión...")
    
    # Definimos la distribución de los parámetros para la búsqueda aleatoria
    param_dist = {
        'max_depth': [3, 5, 7, 10, None], 
        'min_samples_split': [2, 5, 10, 20]
    }
    
    # Creamos el objeto de búsqueda aleatoria
    # n_iter controla el número de combinaciones a probar
    random_search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42), 
        param_distributions=param_dist, 
        n_iter=10,  # Probamos con 10 combinaciones aleatorias
        cv=5, 
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Mejores hiperparámetros encontrados: {random_search.best_params_}")
    return random_search.best_estimator_

def train_neural_network(X_train, y_train, X_test):
    """
    Entrenamineto del modelo de Red Neuronal.
    
    Args:
        X_train (np.array): Conjunto de entrenamiento de características.
        y_train (np.array): Conjunto de entrenamiento de etiquetas.
        X_test (np.array): Conjunto de prueba de características.
        
    Returns:
        tuple: Una tupla con el modelo de Red Neuronal entrenado, el conjunto
               de prueba escalado, y el historial de entrenamiento.
    """
    # Escalar los datos para la Red Neuronal
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Construir el modelo de Red Neuronal
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Iniciando el entrenamiento de la Red Neuronal...")
    history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
    
    return model, X_test_scaled, history.history