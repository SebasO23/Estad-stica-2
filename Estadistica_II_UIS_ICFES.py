import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# --- 1. CARGA DE DATOS ---
try:
  datos_icfes = pd.read_csv("SB11_20231.TXT", delimiter="¬", engine="python")
  print(f"Dimensiones del dataset original: {datos_icfes.shape}")
except Exception as e:
  print("Asegúrate de tener el archivo SB11_20231.TXT en el mismo directorio.", str(e))

# --- 2. LIMPIEZA DE DATOS (VALORES FALTANTES) ---
cols_numericas = [
    "ESTU_INSE_INDIVIDUAL",
    "PUNT_LECTURA_CRITICA",
    "PUNT_MATEMATICAS",
    "PUNT_C_NATURALES",
    "PUNT_SOCIALES_CIUDADANAS",
    "PUNT_INGLES",
]

for col in cols_numericas:
  if col in datos_icfes.columns:
    datos_icfes[col] = pd.to_numeric(datos_icfes[col], errors="coerce")

icfes_sin_nan = datos_icfes.dropna(subset=cols_numericas).copy()
print(f"Registros limpios sin valores faltantes: {icfes_sin_nan.shape[0]}")

icfes_connan = datos_icfes[
    (datos_icfes["ESTU_INSE_INDIVIDUAL"].isna())
    & (~datos_icfes["PUNT_LECTURA_CRITICA"].isna())
    & (~datos_icfes["PUNT_MATEMATICAS"].isna())
    & (~datos_icfes["PUNT_C_NATURALES"].isna())
    & (~datos_icfes["PUNT_SOCIALES_CIUDADANAS"].isna())
    & (~datos_icfes["PUNT_INGLES"].isna())
].copy()

# --- 3. SELECCIÓN DE VARIABLES Y CORRELACIONES ---
variables_explicativas = [
    "PUNT_LECTURA_CRITICA",
    "PUNT_MATEMATICAS",
    "PUNT_C_NATURALES",
    "PUNT_SOCIALES_CIUDADANAS",
    "PUNT_INGLES",
]

subset_corr = icfes_sin_nan[["ESTU_INSE_INDIVIDUAL"] + variables_explicativas]
print("\n--- Coeficientes de Correlación de Pearson ---")
print(subset_corr.corr(method="pearson")[["ESTU_INSE_INDIVIDUAL"]])

# --- 4. MODELOS PREDICTIVOS (TRAIN / TEST) ---
X = icfes_sin_nan[variables_explicativas]
y = icfes_sin_nan["ESTU_INSE_INDIVIDUAL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Regresión Lineal Múltiple
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\n--- Regresión Lineal Múltiple ---")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lr):.4f}")

# Perceptrón Multicapa (MLPRegressor)
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 100, 100, 100, 100, 100),
    max_iter=500,
    activation="relu",
    solver="adam",
    learning_rate="invscaling",
    random_state=42,
)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
print("\n--- Perceptrón Multicapa (MLP) ---")
print(f"MSE: {mean_squared_error(y_test, y_pred_mlp):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_mlp):.4f}")

# --- 5. IMPUTACIÓN DE DATOS FALTANTES ---
mejor_modelo = mlp_model if r2_score(y_test, y_pred_mlp) > r2_score(y_test, y_pred_lr) else lr_model
icfes_connan["ESTU_INSE_INDIVIDUAL_IMPUTADO"] = mejor_modelo.predict(
    icfes_connan[variables_explicativas]
)
print(
    f"\nImputación completada para {len(icfes_connan)} registros faltantes de"
    " INSE."
)
