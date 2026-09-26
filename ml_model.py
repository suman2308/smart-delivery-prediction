"""Machine learning utilities for CourierAI.

Provides:
- A preprocessing + feature engineering pipeline.
- Several regression candidates (LinearRegression, RandomForest, XGBoost, LightGBM).
- Automatic training, validation, and best‑model selection.
- Persistence of the chosen pipeline (joblib) and a tiny JSON metadata file indicating the selected model.
- Prediction and evaluation helpers used by the Flask app (the app never knows which model is chosen).

The training pipeline is completely independent from Flask – it can be invoked from a script or a CI job.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Optional imports – if not available we simply skip those models.
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # type: ignore
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None  # type: ignore

import config
import database as db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = ["distance", "order_time", "traffic_level", "weather"]
TARGET = "delivery_time"

MODEL_PATH = os.environ.get(
    "DELIVERY_MODEL_PATH", os.path.join(config.BASE_DIR, "models", "delivery_regressor.joblib")
)
# Path for a tiny JSON file that records which model was selected.
MODEL_META_PATH = MODEL_PATH + ".meta.json"

# ---------------------------------------------------------------------------
# Pre‑processing / feature engineering
# ---------------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that:
    * Scales numeric columns with ``StandardScaler``.
    * One‑hot encodes categorical columns, ignoring unknown categories.
    """
    numeric = ["distance", "order_time"]
    categorical = ["traffic_level", "weather"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

# ---------------------------------------------------------------------------
# Model registry – candidate regressors
# ---------------------------------------------------------------------------


def _candidate_models() -> Dict[str, Any]:
    """Dictionary of model name → instantiated regressor.
    Models that cannot be imported are silently omitted – the training loop will only
    consider the available ones.
    """
    candidates: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42)
    if LGBMRegressor is not None:
        candidates["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=42)
    return candidates

# ---------------------------------------------------------------------------
# Training result dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    mae: float
    rmse: float
    train_rows: int
    best_model: str

# ---------------------------------------------------------------------------
# Core training routine – independent from Flask
# ---------------------------------------------------------------------------
def train_and_save(test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    """Train all candidate models, evaluate on a hold‑out set, and persist the best.

    Returns a ``TrainResult`` containing the validation MAE/RMSE, the number of rows
    used for training and the name of the selected model.
    """
    rows = db.fetch_orders_for_training()
    if len(rows) < 10:
        raise ValueError("At least 10 historical orders are required to train model.")

    df = db.rows_to_dataframe(rows)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = _build_preprocessor()
    candidates = _candidate_models()

    best_mae = float("inf")
    best_pipe: Optional[Pipeline] = None
    best_name = ""

    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        mae = float(mean_absolute_error(y_val, preds))
        if mae < best_mae:
            best_mae = mae
            best_pipe = pipe
            best_name = name

    if best_pipe is None:
        raise RuntimeError("No candidate models were successfully trained.")

    # Persist the best pipeline.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    # Persist metadata (selected model name).
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name}, f)

    # Compute final metrics on the validation set for reporting.
    final_preds = best_pipe.predict(X_val)
    mae = float(mean_absolute_error(y_val, final_preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, final_preds)))
    return TrainResult(mae=mae, rmse=rmse, train_rows=len(df), best_model=best_name)

# ---------------------------------------------------------------------------
# Load pipeline – Flask only calls this; it never knows which model was chosen.
# ---------------------------------------------------------------------------
def load_pipeline() -> Pipeline:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not initialized at {MODEL_PATH}. Train model via CLI or UI."
        )
    return joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Prediction helper – used by Flask routes.
# ---------------------------------------------------------------------------
def predict_delivery(
    distance: float,
    order_time: int,
    traffic_level: str,
    weather: str,
    pipeline: Optional[Pipeline] = None,
) -> float:
    pipe = pipeline or load_pipeline()
    X = pd.DataFrame([
        {
            "distance": distance,
            "order_time": order_time,
            "traffic_level": traffic_level,
            "weather": weather,
        }
    ])
    out = pipe.predict(X)
    return float(np.maximum(out[0], 2.0))

