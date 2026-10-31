"""
entrenar_pipeline.py

Pipeline de preprocesamiento + modelo para predecir la produccion diaria
de petroleo (BORE_OIL_VOL) de los pozos del campo Volve.
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUTA_DATOS = "data/volve_produccion.csv"
RUTA_MODELO = "modelos/pipeline_volve.joblib"

TARGET = "BORE_OIL_VOL"
COLUMNAS_NUMERICAS = [
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_DP_TUBING",
    "AVG_ANNULUS_PRESS",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "AVG_WHT_P",
    "DP_CHOKE_SIZE",
]
COLUMNAS_CATEGORICAS = ["NPD_WELL_BORE_NAME"]


def cargar_datos(ruta=RUTA_DATOS):
    """Carga el CSV y devuelve X (variables predictoras) e y (target)."""
    df = pd.read_csv(ruta)
    df = df[df["FLOW_KIND"] == "production"]  # en inyeccion no hay petroleo
    df = df.dropna(subset=[TARGET])
    X = df[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS]
    y = df[TARGET]
    return X, y


def dividir_datos(X, y):
    """
    Division train/test. Va ANTES de ajustar cualquier transformacion,
    para que el preprocesamiento no "vea" los datos de test (data leakage).
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def construir_pipeline():
    """Arma el preprocesamiento (ColumnTransformer) + el modelo en un Pipeline."""
    # Numericas: completar nulos con la mediana y escalar
    numericas = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
    ])

    # Categoricas: completar nulos con el valor mas frecuente y codificar.
    # handle_unknown="ignore" evita errores si aparece un pozo nuevo.
    categoricas = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocesador = ColumnTransformer([
        ("num", numericas, COLUMNAS_NUMERICAS),
        ("cat", categoricas, COLUMNAS_CATEGORICAS),
    ])

    return Pipeline([
        ("preprocesamiento", preprocesador),
        ("modelo", LinearRegression()),
    ])


def evaluar(pipeline, X_test, y_test):
    """Predice sobre el conjunto de test e imprime las metricas."""
    pred = pipeline.predict(X_test)
    print(f"MAE : {mean_absolute_error(y_test, pred):.1f}")
    print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.1f}")
    print(f"R2  : {r2_score(y_test, pred):.3f}")


def guardar_pipeline(pipeline, ruta=RUTA_MODELO):
    """Exporta el pipeline entrenado (preprocesamiento + modelo) con joblib."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    joblib.dump(pipeline, ruta)
    print(f"Pipeline guardado en {ruta}")


def main():
    X, y = cargar_datos()
    print(f"Filas: {len(X)} | nulos en X: {int(X.isna().sum().sum())}")

    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)  # se ajusta SOLO con train

    evaluar(pipeline, X_test, y_test)
    guardar_pipeline(pipeline)


if __name__ == "__main__":
    main()
