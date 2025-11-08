import pandas as pd

def make_prediction(model, feature_order, threshold, input_dict):
    # Convertir entrada a DataFrame con el orden correcto
    df = pd.DataFrame([input_dict])[feature_order]

    # Probabilidad de clase positiva
    proba = model.predict_proba(df)[:,1][0]

    # Predicción binaria con threshold
    pred = int(proba >= threshold)

    return pred, float(proba)
