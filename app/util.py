import json
from xgboost import XGBClassifier

def load_artifacts():
    model = XGBClassifier()
    model.load_model("ml/modelo_xgb.json")

    with open("ml/feature_order.json", "r") as f:
        feature_order = json.load(f)

    with open("ml/modelo_meta.json", "r") as f:
        meta = json.load(f)

    return model, feature_order, meta