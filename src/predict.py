import joblib
import pandas as pd

from src.features import build_features

MODEL_PATH = "model/gbm.pkl"

model = joblib.load(MODEL_PATH)

def predict_no_show(payload: dict):
    raw_df = pd.DataFrame([payload])
    features_df = build_features(raw_df)

    if "no_show" in features_df.columns:
        features_df = features_df.drop(columns=["no_show"])

    prob = model.predict_proba(features_df)[0, 1]

    return {
        "no_show_probability": round(float(prob), 4),
        "prediction": "Likely no-show" if prob >= 0.5 else "Likely show",
        "model_used": "gradient_boosting"
    }
