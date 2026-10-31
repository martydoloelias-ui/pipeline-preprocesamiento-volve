"""
predecir.py

Carga el pipeline entrenado con joblib y predice sobre datos nuevos.
"""

import joblib
import pandas as pd

RUTA_MODELO = "modelos/pipeline_volve.joblib"


def main():
    pipeline = joblib.load(RUTA_MODELO)

    nuevos = pd.DataFrame([
        {   # pozo conocido, con todos los datos
            "ON_STREAM_HRS": 24,
            "AVG_DOWNHOLE_PRESSURE": 250.0,
            "AVG_DOWNHOLE_TEMPERATURE": 105.0,
            "AVG_DP_TUBING": 120.0,
            "AVG_ANNULUS_PRESS": 20.0,
            "AVG_CHOKE_SIZE_P": 60.0,
            "AVG_WHP_P": 40.0,
            "AVG_WHT_P": 80.0,
            "DP_CHOKE_SIZE": 15.0,
            "NPD_WELL_BORE_NAME": "15/9-F-12",
        },
        {   # pozo que el modelo nunca vio y con datos faltantes
            "ON_STREAM_HRS": 24,
            "AVG_DOWNHOLE_PRESSURE": None,
            "AVG_DOWNHOLE_TEMPERATURE": None,
            "AVG_DP_TUBING": None,
            "AVG_ANNULUS_PRESS": None,
            "AVG_CHOKE_SIZE_P": 50.0,
            "AVG_WHP_P": 35.0,
            "AVG_WHT_P": 75.0,
            "DP_CHOKE_SIZE": 10.0,
            "NPD_WELL_BORE_NAME": "15/9-F-99",
        },
    ])

    predicciones = pipeline.predict(nuevos)
    for pozo, valor in zip(nuevos["NPD_WELL_BORE_NAME"], predicciones):
        print(f"{pozo}: {valor:.1f}")


if __name__ == "__main__":
    main()