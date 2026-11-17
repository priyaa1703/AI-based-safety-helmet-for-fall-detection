"""
preprocess.py
-------------
Cleaning, encoding, splitting and scaling.

Order matters here to avoid data leakage:
  1. mark impossible values as missing (row-by-row rule, no statistics needed)
  2. split into train / test  (stratified)
  3. fit the imputer + scaler on TRAIN only
  4. apply them to train and test

Run directly to produce the artefacts the other scripts use:
    python preprocess.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_CSV = ROOT / "data" / "helmet_sensor_dataset.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# This order is used everywhere, including the ESP32 code. Do not change it
# without regenerating everything.
FEATURES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
            "HeartRate", "SpO2", "Temperature", "Humidity"]
CLASSES = ["NORMAL", "FALL", "HEALTH_ALERT", "ENVIRONMENT_ALERT"]   # index = class id
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Physically possible range for each sensor. Anything outside is a glitch.
VALID_RANGE = {
    "AccX": (-16, 16), "AccY": (-16, 16), "AccZ": (-16, 16),     # g
    "GyroX": (-2000, 2000), "GyroY": (-2000, 2000), "GyroZ": (-2000, 2000),
    "HeartRate": (25, 220),     # bpm
    "SpO2": (70, 100),          # % (below 70 the MAX30102 is not trustworthy)
    "Temperature": (-10, 70),   # deg C
    "Humidity": (0, 100),       # %
}

TEST_SIZE = 0.20
SEED = 42


def load_raw(path=DATA_CSV):
    return pd.read_csv(path)


def check_data(df):
    """Print a short data-quality report and return count of invalid values."""
    print("Shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nInvalid (out-of-range) values per column:")
    invalid_total = {}
    for col, (lo, hi) in VALID_RANGE.items():
        bad = ((df[col] < lo) | (df[col] > hi)).sum()
        invalid_total[col] = int(bad)
    print(pd.Series(invalid_total).to_string())

    dup = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup}")

    print("\nClass distribution:")
    counts = df["Label"].value_counts()
    print(pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(1)}).to_string())
    return invalid_total


def mark_invalid_as_nan(df):
    df = df.copy()
    for col, (lo, hi) in VALID_RANGE.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def prepare_data(verbose=True, save=False):
    """
    Returns a dict with scaled arrays and the fitted objects.
    Keys: X_train, X_test, y_train, y_test (scaled / integer-encoded),
          X_train_raw, X_test_raw (imputed but not scaled), imputer, scaler
    """
    df = load_raw()
    if verbose:
        check_data(df)

    df = df.drop_duplicates().reset_index(drop=True)
    df = mark_invalid_as_nan(df)

    # rows with no label can't be used
    df = df.dropna(subset=["Label"])
    y = df["Label"].map(CLASS_TO_ID).astype(int).values
    X = df[FEATURES].values.astype(np.float32)

    # stratified 80/20 split BEFORE any statistics are computed
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    # median is robust to the odd outlier, and is fitted on train only
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp).astype(np.float32)
    X_te_s = scaler.transform(X_te_imp).astype(np.float32)

    if verbose:
        print(f"\nTrain: {X_tr_s.shape}  Test: {X_te_s.shape}")
        print("Train class counts:", np.bincount(y_tr).tolist())
        print("Test  class counts:", np.bincount(y_te).tolist())

    out = dict(X_train=X_tr_s, X_test=X_te_s, y_train=y_tr, y_test=y_te,
               X_train_raw=X_tr_imp, X_test_raw=X_te_imp,
               imputer=imputer, scaler=scaler)

    if save:
        MODELS.mkdir(exist_ok=True)
        joblib.dump(scaler, MODELS / "scaler.pkl")
        joblib.dump(imputer, MODELS / "imputer.pkl")
        # plain json copy: this is what gets typed into the ESP32 code
        params = {
            "feature_order": FEATURES,
            "class_order": CLASSES,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),          # = standard deviation
            "fallback_values_for_missing": imputer.statistics_.tolist(),
        }
        (MODELS / "scaler_params.json").write_text(json.dumps(params, indent=2))
        print("\nSaved scaler.pkl, imputer.pkl, scaler_params.json in models/")
    return out


if __name__ == "__main__":
    prepare_data(verbose=True, save=True)
