# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import json
import pandas as pd
from xgboost import XGBClassifier
from app.explain import explain_prediction

import shap

import numpy as np
import xgboost as xgb


# === 1. Inicializar API ===
app = FastAPI(title="Coronary Risk API", version="1.0")

# === 2. Cargar artefactos ===
model = XGBClassifier()
model.load_model("ml/modelo_xgb.json")

with open("ml/feature_order.json", "r") as f:
    feature_order = json.load(f)

with open("ml/modelo_meta.json", "r") as f:
    meta = json.load(f)

threshold = meta["threshold_usado"]

# === 3. Definir esquema de entrada ===
class UserData(BaseModel):
    Diabetes_012: int
    Hipertension: int
    Colesterol_alto: int
    Chequeo_colesterol: int
    IMC: float
    Fumador: int
    Derrame_cerebral: int
    Actividad_fisica: int
    Frutas: int
    Verduras: int
    Consumo_alcohol_excesivo: int
    Cobertura_salud: int
    No_visita_medico_por_costo: int
    Salud_general: int
    Salud_mental_dias_malos: int
    Salud_fisica_dias_malos: int
    Dificultad_caminar: int
    Sexo: int
    Edad: int
    Educacion: int

# === 4. Endpoint de predicción ===
@app.post("/predict")
def predict(data: UserData):
    df = pd.DataFrame([data.dict()])[feature_order]
    proba = float(model.predict_proba(df)[:, 1][0])
    pred = int(proba >= threshold)
    return {"prediccion": pred, "probabilidad": round(proba, 4)}

# === 5. Endpoint de explicación ===
@app.post("/explain")
def explain(data: UserData):
    try:
        # Construir DataFrame con el orden correcto
        df = pd.DataFrame([data.dict()])[feature_order]

        # Predicción y probabilidad
        proba = float(model.predict_proba(df)[:, 1][0])
        pred = int(proba >= threshold)

        # SHAP nativo vía XGBoost
        dmatrix = xgb.DMatrix(df)
        shap_values = model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_row = shap_values[0, :-1]  # última columna es el bias

        # Seleccionar top 3 variables más influyentes
        top_idx = np.argsort(np.abs(shap_row))[::-1][:3]
        explicacion = []
        for i in top_idx:
            var = feature_order[i]
            impacto = float(shap_row[i])
            direccion = "aumenta el riesgo" if impacto > 0 else "reduce el riesgo"
            explicacion.append({
                "variable": str(var),
                "impacto": round(abs(impacto), 4),
                "direccion": direccion
            })

        # Respuesta JSON‑friendly
        return {
            "prediccion": pred,
            "probabilidad": round(proba, 4),
            "explicacion": explicacion
        }
    except Exception as e:
        return {"error": str(e)}