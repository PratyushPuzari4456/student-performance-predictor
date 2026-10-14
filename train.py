import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "semester",
    "current_cgpa",
    "last_sgpa",
    "backlogs",
    "attendance_percent",
    "internal_marks",
    "assignment_avg",
    "study_hours_per_week",
    "sleep_hours_per_night",
    "internet_hours_per_day",
    "motivation_level",
    "family_support_level",
    "part_time_job",
]
TARGET = "final_percentage"


def generate_sample_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    df["semester"] = rng.integers(1, 9, n)
    df["current_cgpa"] = np.clip(rng.normal(7.0, 1.2, n), 4.0, 10.0).round(2)
    df["last_sgpa"] = np.clip(df["current_cgpa"] + rng.normal(0, 0.5, n), 3.5, 10.0).round(2)
    df["backlogs"] = rng.integers(0, 4, n)
    df["attendance_percent"] = np.clip(rng.normal(78, 12, n), 30, 100).round(1)
    df["internal_marks"] = np.clip(rng.normal(65, 15, n), 20, 100).round(1)
    df["assignment_avg"] = np.clip(rng.normal(70, 14, n), 20, 100).round(1)
    df["study_hours_per_week"] = np.clip(rng.normal(14, 6, n), 0, 45).round(1)
    df["sleep_hours_per_night"] = np.clip(rng.normal(7, 1.2, n), 4, 10).round(1)
    df["internet_hours_per_day"] = np.clip(rng.normal(3, 1.5, n), 0, 10).round(1)
    df["motivation_level"] = rng.integers(1, 6, n)
    df["family_support_level"] = rng.integers(1, 6, n)
    df["part_time_job"] = rng.integers(0, 2, n)

    score = (
        2.2 * df["current_cgpa"]
        + 1.0 * df["last_sgpa"]
        + 0.22 * df["attendance_percent"]
        + 0.18 * df["internal_marks"]
        + 0.13 * df["assignment_avg"]
        + 0.50 * df["study_hours_per_week"]
        - 2.2 * df["backlogs"]
        + 1.2 * df["motivation_level"]
        + 0.9 * df["family_support_level"]
        - 1.5 * df["part_time_job"]
        + 0.6 * df["sleep_hours_per_night"]
        - 0.9 * df["internet_hours_per_day"]
        + rng.normal(0, 4, n)
    )
    df[TARGET] = np.clip(score, 30, 100).round(1)
    return df


def main():
    data_path = DATA_DIR / "students.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            print(f"Real CSV missing columns {missing}. Using generated sample data.")
            df = generate_sample_data()
        else:
            print("Using real data from data/students.csv")
    else:
        print("No real data found. Creating sample data...")
        df = generate_sample_data()
        df.to_csv(data_path, index=False)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Training completed. Rows: {len(df)} | MAE: {mae:.2f} | R2: {r2:.2f}")

    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "columns.json").write_text(json.dumps(FEATURES), encoding="utf-8")

    medians = df[FEATURES].median().round(2).to_dict()
    metrics = {
        "rows": int(len(df)),
        "mae": round(float(mae), 2),
        "r2": round(float(r2), 2),
        "medians": medians,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    print("Model saved to models/model.joblib")


if __name__ == "__main__":
    main()