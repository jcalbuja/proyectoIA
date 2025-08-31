import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess_data(df):
    """
    Preprocesamos los datos, eliminando columnas innecesarias, 
    manejando valores infinitos y nulos, y codificando la variable objetivo.
    
    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        
    Returns:
        tuple: Tupla que contiene las características (X), la variable objetivo (y),
               y el codificador de etiquetas.
    """
    # Eliminamos columnas con valores infinitos o que no aportan al modelo
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Eliminamos las columna 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
    # 'Timestamp' ya que no son características útiles para la clasificación
    df = df.drop(columns=['Flow ID', ' Source IP', ' Source Port',
                          ' Destination IP', ' Timestamp'], errors='ignore')

    # Separamos las características (X) de la variable objetivo (y)
    X = df.drop(columns=[' Label'])
    y = df[' Label']
    
    # Codificamos la variable objetivo 'Label' (BENIGN y DDoS)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y, label_encoder