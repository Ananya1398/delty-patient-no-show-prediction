import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data import load_data
from src.features import build_features

DATA_PATH = "data/appointments.csv"
MODEL_DIR = "model/"

def train():
    df = load_data(DATA_PATH)
    df = build_features(df, training = True)

    X = df.drop(columns=["no_show"])
    y = df["no_show"]

    categorical = ["Gender", "Neighbourhood", "appt_dow", "appt_month", "scheduled_dow"]
    numeric = ["Age", "lead_time_days", "scheduled_hour", "Handicapped", "comorbidity_count"]
    binary = [
        "OnGovtWelfareBenefits", "Hypertension", "Diabetes", "Alcoholism",
        "SMS_received", "has_chronic", "has_disability", "is_same_day"
    ]

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
            ("bin", "passthrough", binary),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Logistic Regression
    logreg = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])
    logreg.fit(X_train, y_train)

    # Gradient Boosting
    gbm = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss"
        ))
    ])
    gbm.fit(X_train, y_train)

    # Save
    joblib.dump(logreg, MODEL_DIR + "logreg.pkl")
    joblib.dump(gbm, MODEL_DIR + "gbm.pkl")
    joblib.dump(preprocessor, MODEL_DIR + "preprocessor.pkl")

    print("Models saved successfully.")

if __name__ == "__main__":
    train()
