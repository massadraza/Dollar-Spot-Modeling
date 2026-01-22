import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def xgboost_model_FEATURE_ENGINEERED_LOG():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    feature_cols = [
        "Rating", "Date", "Year", "Season", "Obs", "Rep",
        "meanLW", "meanST", "meanSM", "meanRH", "meanAT", "meanRF",
        "maxLW", "maxST", "maxSM", "maxRH", "maxAT", "maxRF",
        "minST", "minSM", "minRH", "minAT"
    ]

    X = df[feature_cols].copy()
    y = df["Foci"]
    y_log = np.log1p(y)  # log-transform target

    X['rangeRH'] = X['maxRH'] - X['minRH']
    X['rangeSM'] = X['maxSM'] - X['minSM']
    X['rangeAT'] = X['maxAT'] - X['minAT']

    X['SM_AT'] = X['meanSM'] * X['meanAT']
    X['RH_AT'] = X['meanRH'] * X['meanAT']
    X['SM_RH'] = X['meanSM'] * X['meanRH']

    categorical_cols = ["Season"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough"
    )

    xgb_model = Pipeline([
        ("preprocess", preprocess),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror'
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    xgb_model.fit(X_train, y_train)

    y_pred_log = xgb_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("=========== XGBoost METRICS (FEATURE ENGINEERED + LOG) ===========")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

xgboost_model_FEATURE_ENGINEERED_LOG()

"""
=========== XGBoost METRICS (FEATURE ENGINEERED + LOG) ===========
RMSE: 4.2564
MAE:  1.5794
R²:   0.6008

"""