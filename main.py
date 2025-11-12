# main.py
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

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

def limpiar_numpy(obj):
    if isinstance(obj, dict):
        return {k: limpiar_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [limpiar_numpy(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj



@app.post("/explain")
def explain(data: UserData):
    try:
        resultado = explain_prediction(
            model=model,
            feature_order=feature_order,
            threshold=threshold,
            input_dict=data.dict(),
            min_pct=10
        )
        # ✅ Limpieza y serialización segura
        return JSONResponse(content=jsonable_encoder(limpiar_numpy(resultado)))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
