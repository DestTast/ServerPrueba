import numpy as np
import pandas as pd
import xgboost as xgb

def explain_prediction(model, feature_order, threshold, input_dict, min_pct=10):
    # Convertir entrada a DataFrame con el orden correcto
    df = pd.DataFrame([input_dict])[feature_order]

    # Probabilidad y predicción
    proba = float(model.predict_proba(df)[:, 1][0])
    pred = int(proba >= threshold)

    # SHAP nativo vía XGBoost
    dmatrix = xgb.DMatrix(df)
    shap_values = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_row = shap_values[0, :-1]  # última columna es el bias/base_value

    # === 1. Calcular impactos y porcentajes ===
    total_impacto = sum(abs(val) for val in shap_row)
    impactos = []
    for i, valor in enumerate(shap_row):
        impacto_abs = abs(float(valor))
        porcentaje = (impacto_abs / total_impacto) * 100 if total_impacto > 0 else 0
        direccion = "aumenta el riesgo" if valor > 0 else "reduce el riesgo"
        impactos.append({
            "variable": feature_order[i],
            "impacto": round(impacto_abs, 4),
            "porcentaje": round(porcentaje, 2),
            "direccion": direccion
        })

    # === 2. Filtrar las más influyentes (ej. ≥10%) ===
    influyentes = [v for v in impactos if v["porcentaje"] >= min_pct]
    influyentes_ordenadas = sorted(influyentes, key=lambda x: -x["porcentaje"])

    # === 3. Respuesta JSON ===
    return {
        "prediccion": pred,
        "probabilidad": round(proba, 4),
        "impactos": impactos,  # todas las variables
        "variables_influyentes": influyentes_ordenadas  # solo las que aportan más
    }
