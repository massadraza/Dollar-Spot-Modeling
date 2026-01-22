import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def gradient_boosting_baselineModel():
    
    df = pd.read_csv("Shoulder_Season_Data.csv")

    features = [
        'meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT',
        'maxLW', 'maxSM', 'maxRH', 'maxAT', 'maxRF',
        'minST', 'minSM', 'minRH', 'minAT'
    ]

    X = df[features].dropna()
    y = df.loc[X.index, 'Foci']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)


    print("======= GRADIENT BOOSTING RAW BASELINE =======")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

#gradient_boosting_baselineModel()

def gradient_boosting_baselineModel_log():
    
    df = pd.read_csv("Shoulder_Season_Data.csv")

    features = [
        'meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT',
        'maxLW', 'maxSM', 'maxRH', 'maxAT', 'maxRF',
        'minST', 'minSM', 'minRH', 'minAT'
    ]

    X = df[features].dropna()
    y = df.loc[X.index, 'Foci']
    y_log = np.log1p(y)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)


    print("======= GRADIENT BOOSTING Log Transformation =======")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=True)


    """
    print("\n Feature Importances: ")
    print(importances)
    """
#gradient_boosting_baselineModel_log()

# BEST MODEL
def gradient_boosting_model_encoded():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    feature_cols = [
        "Rating", "Date", "Year", "Season", "Obs", "Rep",
        "meanLW", "meanST", "meanSM", "meanRH", "meanAT", "meanRF",
        "maxLW", "maxST", "maxSM", "maxRH", "maxAT", "maxRF",
        "minST", "minSM", "minRH", "minAT"
    ]

    X = df[feature_cols]
    y = df["Foci"]

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
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=========== GRADIENT BOOSTING METRICS ===========")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

#gradient_boosting_model_encoded()

def gradient_boosting_model_encoded_log():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    feature_cols = [
        "Rating", "Date", "Year", "Season",
        "meanLW", "meanST", "meanSM", "meanRH", "meanAT", "meanRF",
        "maxLW", "maxST", "maxSM", "maxRH", "maxAT", "maxRF",
        "minST", "minSM", "minRH", "minAT"
    ]

    X = df[feature_cols]
    y = df["Foci"]

    y_log = np.log1p(y)

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

    print("=========== GRADIENT BOOSTING METRICS (LOG TRANSFORMATION) ===========")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

#gradient_boosting_model_encoded_log()  

def gradient_boosting_model_encoded_log_TUNED():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    feature_cols = [
        "Rating", "Date", "Year", "Season", "Obs", "Rep",
        "meanLW", "meanST", "meanSM", "meanRH", "meanAT", "meanRF",
        "maxLW", "maxST", "maxSM", "maxRH", "maxAT", "maxRF",
        "minST", "minSM", "minRH", "minAT"
    ]

    X = df[feature_cols]
    y = df["Foci"]

    y_log = np.log1p(y)

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

    param_grid = {
        'model__n_estimators': [200, 300, 500],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [2, 3, 4],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__max_features': [None, 'sqrt', 'log2']
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
   
    grid_search = GridSearchCV(
        model,
        param_grid, 
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    y_pred_log = grid_search.best_estimator_.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("=========== GRADIENT BOOSTING METRICS (LOG + GRIDSEARCHCV) ===========")
    print(f"Best MAE (CV): {-grid_search.best_score_:.4f}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"RMSE on Test: {rmse:.4f}")
    print(f"MAE on Test:  {mae:.4f}")
    print(f"R² on Test:   {r2:.4f}")



def gradient_boosting_model_encoded_FEATURE_ENGINEERED():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    feature_cols = [
        "Rating", "Date", "Year", "Season", "Obs", "Rep",
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

#gradient_boosting_model_encoded_FEATURE_ENGINEERED()

#gradient_boosting_baselineModel()
#gradient_boosting_baselineModel_log()
#gradient_boosting_model_encoded()
#gradient_boosting_baselineModel_log()
#gradient_boosting_model_encoded_log()
#gradient_boosting_model_encoded_log_TUNED()
gradient_boosting_model_encoded_FEATURE_ENGINEERED() # ---> New Best Model So Far

"""
Gradient Boosting Model Encoded Log Is the Best ML Model Thus Far

=========== GRADIENT BOOSTING METRICS (LOG TRANSFORMATION) ===========
RMSE: 4.0777
MAE:  1.5441
R²:   0.6336

"""

"""
=========== GRADIENT BOOSTING METRICS (FEATURE ENGINEERED) ===========
RMSE: 3.9065
MAE:  1.5021
R²:   0.6637
"""