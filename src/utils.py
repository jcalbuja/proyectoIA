import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Realiza el preprocesamiento del dataset.
    - Limpia valores infinitos y nulos.
    - Codifica la variable objetivo (Label).
    - Separa las características (X) de las etiquetas (y).
    """
    # Renombrar la columna ' Label' a 'Label' para mayor comodidad
    df.rename(columns={' Label': 'Label'}, inplace=True)
    
    # Reemplazar valores infinitos con NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Llenar los valores NaN con el promedio de la columna
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Definir características (X) y variable objetivo (y)
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Codificar la variable objetivo 'Label' a valores numéricos (0 y 1)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le