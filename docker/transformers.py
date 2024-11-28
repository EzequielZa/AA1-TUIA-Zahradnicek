# Función de preprocesamiento
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, OrdinalEncoder

def preprocess_data(df):

    cuidades_aleatorias = ['Mildura', 'Sale', 'Brisbane', 'Melbourne', 'Sydney', 'Moree', 'NorahHead', 'Portland', 'Ballarat', 'Albany']
    df = df[df['Location'].isin(cuidades_aleatorias)]
    df = df.drop('Location', axis=1)

    # Codificar la columna 'Date'
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df = df.drop(columns=['Date', 'day_of_year'])

    # Codificar las direcciones del viento
    order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    encoder = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=np.nan)
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        df[col] = encoder.fit_transform(df[[col]])
        df[f'{col}_sin'] = np.sin(np.deg2rad(df[col] * 22.5))
        df[f'{col}_cos'] = np.cos(np.deg2rad(df[col] * 22.5))
        df = df.drop(columns=[col])

    # Codificar 'RainToday'
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})

    return df

def imputer(df):

    columns = df.columns
    knn_imputer = KNNImputer(n_neighbors=5)

    df = knn_imputer.fit_transform(df)
    df = pd.DataFrame(df, columns=columns)

    return df

# Función de escalado
def Scaler(df):

    # Identificar columnas para escalado
    columnas_binarias = [col for col in df.columns if df[col].nunique() == 2 and sorted(df[col].unique()) == [0, 1]]
    columnas_trigonometricas = [col for col in df.columns if 'sin' in col or 'cos' in col]
    columnas_imputables = df.columns.difference(columnas_binarias).difference(columnas_trigonometricas)

    power_transformer = PowerTransformer(method='yeo-johnson')
    scaler = StandardScaler()

    # Escalar las columnas identificadas
    df[columnas_imputables] = power_transformer.fit_transform(df[columnas_imputables])
    df[columnas_imputables] = scaler.fit_transform(df[columnas_imputables])

    return df



# Transformer para preprocesamiento
class PreprocessDataTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return preprocess_data(X)

# Imputer KNN
class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return imputer(X)

# Transformer para escalado
class ScaleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return Scaler(X)