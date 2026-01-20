import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

# Required to remove obs and rep --> Seen as an overfitting issue

def gradient_boosting_model_encoded_FEATURE_ENGINEERED():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    feature_cols = [
        "Rating", "Date", "Year", "Season",
        "meanLW", "meanST", "meanSM", "meanRH", "meanAT", "meanRF",
        "maxLW", "maxST", "maxSM", "maxRH", "maxAT", "maxRF",
        "minST", "minSM", "minRH", "minAT"
    ]

    X = df[feature_cols].copy()
    y = df["Foci"]
    y_log = np.log1p(y)

    X['rangeRH'] = X['maxRH'] - X['minRH']
    X['rangeSM'] = X['maxSM'] - X['minSM']
    X['rangeAT'] = X['maxAT'] - X['minAT']

    X['SM_AT'] = X['meanSM'] * X['meanAT']
    X['RH_AT'] = X['meanRH'] * X['meanAT']
    X['SM_RH'] = X['meanSM'] * X['meanRH']

    categorical_cols = ["Season"]

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough"
    )
    
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
   
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("=========== GRADIENT BOOSTING METRICS (FEATURE ENGINEERED) ===========")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

gradient_boosting_model_encoded_FEATURE_ENGINEERED()

"""
======= GRADIENT BOOSTING RAW BASELINE =======
MAE:  2.1127
RMSE: 5.2020
R²:   0.4037

"""

"""
With removing the obs and rep, there is a significant reduction
in the accuract of the model. As noted, obs and rep should not 
be providing any acutal meaning in regards to disease pressure
prediction.
"""