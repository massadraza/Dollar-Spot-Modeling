import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# PLS DISREGARD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score


def basic_gradient_boosting_model():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    df['rangeRH'] = df['maxRH'] - df['minRH']
    df['rangeSM'] = df['maxSM'] - df['minSM']
    df['rangeAT'] = df['maxAT'] - df['minAT']

    df['SM_AT'] = df['meanSM'] * df['meanAT']
    df['RH_AT'] = df['meanRH'] * df['meanAT']
    df['SM_RH'] = df['meanSM'] * df['meanRH']

    features = [
        'meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT',
        'maxLW', 'maxSM', 'maxRH', 'maxAT',
        'minST', 'minSM', 'minRH', 'minAT',
        'rangeRH', 'rangeSM', 'rangeAT', 
        'SM_AT', 'RH_AT', 'SM_RH'
    ]

    X = df[features].dropna()
    y = df.loc[X.index, 'Foci']

    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    gb_model = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)

    y_pred_log = gb_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    
    print("===== GRADIENT BOOSTING BASELINE =====")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    
    """
    print(df['Foci'].describe())
    print("Zero proportion:", (df['Foci'] == 0).mean())
    """

basic_gradient_boosting_model()

def gradient_boosting_foci_model():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    df = pd.read_csv("Shoulder_Season_Data.csv")

    # -----------------------------
    # 2. Feature engineering
    # -----------------------------
    df['rangeRH'] = df['maxRH'] - df['minRH']
    df['rangeSM'] = df['maxSM'] - df['minSM']
    df['rangeAT'] = df['maxAT'] - df['minAT']

    df['SM_AT'] = df['meanSM'] * df['meanAT']
    df['RH_AT'] = df['meanRH'] * df['meanAT']
    df['SM_RH'] = df['meanSM'] * df['meanRH']

    # -----------------------------
    # 3. Feature list
    # -----------------------------
    features = [
        'meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT',
        'maxLW', 'maxSM', 'maxRH', 'maxAT', 'maxRF',
        'minST', 'minSM', 'minRH', 'minAT',
        'rangeRH', 'rangeSM', 'rangeAT',
        'SM_AT', 'RH_AT', 'SM_RH'
    ]

    # -----------------------------
    # 4. Prepare X and y (log target)
    # -----------------------------
    X = df[features].dropna()
    y = np.log1p(df.loc[X.index, 'Foci'])

    # -----------------------------
    # 5. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 6. Gradient Boosting model
    # -----------------------------
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    # -----------------------------
    # 7. Train
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # 8. Predict
    # -----------------------------
    y_pred_log = model.predict(X_test)

    # Convert back from log scale
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)

    # -----------------------------
    # 9. Metrics
    # -----------------------------
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
   # rmse = mean_squared_error(y_test_actual, y_pred_actual, squared=False)
    r2 = r2_score(y_test_actual, y_pred_actual)

    print("===== GRADIENT BOOSTING RESULTS =====")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

   

#gradient_boosting_foci_model()