import numpy as np
import pandas as pd
import xgboost as xgb

def explain_prediction(model, feature_order, threshold, input_dict, top_n=3):
    # Convertir entrada a DataFrame con el orden correcto
    df = pd.DataFrame([input_dict])[feature_order]

    # Probabilidad y predicción
    proba = float(model.predict_proba(df)[:, 1][0])
    pred = int(proba >= threshold)

    # SHAP nativo vía XGBoost
    dmatrix = xgb.DMatrix(df)
    shap_values = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_row = shap_values[0, :-1]  # última columna es el bias/base_value

    # Seleccionar top-N variables más influyentes
    top_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
    explicacion = []
    for i in top_idx:
        var = feature_order[i]
        impacto = shap_row[i]
        direccion = "aumenta el riesgo" if impacto > 0 else "reduce el riesgo"
        explicacion.append({
            "variable": var,
            "impacto": round(abs(float(impacto)), 4),
            "direccion": direccion
        })

    return {
        "prediccion": pred,
        "probabilidad": round(proba, 4),
        "explicacion": explicacion
    }
