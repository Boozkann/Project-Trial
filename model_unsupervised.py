# model_unsupervised.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RISK_FEATURES_DEFAULT = [
    "central_govt_debt_gdp","cash_surplus_deficit_gdp","debt_service_to_exports",
    "current_account_gdp","reserves_months_imports","inflation_cpi","gdp_growth",
    "unemployment_rate","m2_gdp","trade_openness","political_stability_estimate",
    "gdp_growth_3y_avg","inflation_vol_5y","debt_x_lendrate",
    "central_govt_debt_gdp_lag1","inflation_cpi_lag1",
    "cash_surplus_deficit_gdp_lag1","gdp_growth_lag1",
    "reserves_months_imports_lag1","debt_gdp_gt_90","reserves_lt_3m"
]

def fit_iforest(X: pd.DataFrame, contamination=0.1):
    # 🧹 güvenlik: NaN temizliği
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True)).dropna(axis=0, how="any")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=300, contamination=contamination, random_state=42
        ))
    ])
    pipe.fit(X)
    pipe.features_ = X.columns.tolist()
    return pipe

def score_risk(pipe, X: pd.DataFrame) -> pd.DataFrame:
    Xp = X[pipe.features_].fillna(X.median(numeric_only=True)).dropna(axis=0, how="any")
    dec = pipe.named_steps["iforest"].decision_function(
        pipe.named_steps["scaler"].transform(Xp)
    )
    import numpy as np
    eps = 1e-9
    risk = 1.0 - (dec - dec.min()) / (dec.max() - dec.min() + eps)
    return pd.DataFrame({"anomaly_score": -dec, "risk_prob": risk}, index=Xp.index)
