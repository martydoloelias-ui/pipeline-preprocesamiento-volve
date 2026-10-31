# Pipeline de preprocesamiento con scikit-learn (campo Volve)

Pipeline reutilizable que prepara los datos de producción del campo
petrolero Volve y entrena un modelo para predecir la producción diaria
de petróleo, evitando la fuga de datos (data leakage).

## Dataset

Producción diaria del campo Volve (Equinor), disponible en
[Kaggle](https://www.kaggle.com/datasets/nazarmahadialseied/volve-field-production-dataset-oil-and-gas).

- 15.634 registros diarios de 7 pozos (2007 a 2016) y 16 columnas.
- Objetivo: predecir `BORE_OIL_VOL` (petróleo producido por día).
- Variables numéricas (con nulos): horas en producción, presiones,
  temperaturas y apertura del choke.
- Variable categórica: `NPD_WELL_BORE_NAME` (nombre del pozo).

Decisiones sobre los datos:
- Se usan solo las filas de producción (9.161), porque en las de
  inyección el petróleo es nulo.
- No se usan como predictoras el gas, el agua ni el agua inyectada:
  son mediciones del mismo día que no estarían disponibles al momento
  de predecir.

## Requisitos

- Python 3.9 o superior.
- Librerías: scikit-learn, pandas y joblib (`requirements.txt`).

## Pasos para ejecutar

1. Clonar el repositorio y entrar a la carpeta:

   ```bash
   git clone <url-del-repo>
   cd pipeline-preprocesamiento-volve
   ```

2. Instalar las dependencias:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Entrenar el pipeline (genera `modelos/pipeline_volve.joblib`):

   ```bash
   python entrenar_pipeline.py
   ```

4. Predecir sobre datos nuevos con el pipeline guardado:

   ```bash
   python predecir.py
   ```

## Qué hace el pipeline

1. **Split**: se divide en 80% train y 20% test **antes** de ajustar
   cualquier transformación, así el preprocesamiento no ve los datos de
   test.
2. **ColumnTransformer**, con un tratamiento distinto por tipo de variable:
   - Numéricas: imputación de nulos con la mediana + `StandardScaler`.
   - Categóricas: imputación con el valor más frecuente + `OneHotEncoder`
     con `handle_unknown="ignore"`, para que un pozo nuevo no genere error.
3. **Pipeline**: encadena el preprocesamiento con una `LinearRegression`
   en un único objeto, que se ajusta con `fit` solo sobre train.
4. **Persistencia**: el pipeline entrenado completo se exporta con
   `joblib`, así la inferencia aplica exactamente el mismo
   preprocesamiento que el entrenamiento.

## Resultados (conjunto de test)

| Métrica | Valor |
|---------|-------|
| MAE     | 525,2 |
| RMSE    | 743,9 |
| R²      | 0,673 |

El modelo lineal explica cerca del 67% de la variación de la producción
diaria. Es una línea base simple, y modelos no lineales podrían mejorarla.

## Estructura

```
.
├── data/volve_produccion.csv   # dataset
├── entrenar_pipeline.py        # split, pipeline, entrenamiento, métricas y joblib
├── predecir.py                 # carga el pipeline y predice datos nuevos
├── requirements.txt
└── README.md
```