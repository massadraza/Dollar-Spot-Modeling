import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report

def random_Forest_Regressor_meanANDmaxANDmin():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    features_mean = ['meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT', 'maxLW', 'maxSM', 
                     'maxRH', 'maxAT', 'maxRF', 'minST', 'minSM', 'minRH', 'minAT']

    X = df[features_mean].dropna()
    Y = df.loc[X.index, 'Foci']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        min_samples_leaf=5
    )

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(Y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
    r2 = r2_score(Y_test, y_pred)
 
    print("======= BASIC METRICS (MEAN and MAX/MIN included) =======")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
   

def random_Forest_Regressor_meanANDmaxANDminV1():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    df['rangeRH'] = df['maxRH'] - df['minRH']
    df['rangeSM'] = df['maxSM'] - df['minSM']
    df['rangeAT'] = df['maxAT'] - df['minAT']

    df['SM_AT'] = df['meanSM'] * df['meanAT']
    df['RH_AT'] = df['meanRH'] * df['meanAT']
    df['SM_RH'] = df['meanSM'] * df['meanRH']
    
    features_mean = ['meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT', 'maxLW', 'maxSM', 
                     'maxRH', 'maxAT', 'maxRF', 'minST', 'minSM', 'minRH', 'minAT',
                     'rangeRH', 'rangeSM', 'rangeAT', 'SM_AT', 'RH_AT', 'SM_RH']

    X = df[features_mean].dropna()
    Y = np.log1p(df.loc[X.index, 'Foci'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        min_samples_leaf=7,
        n_jobs=-1
    )

    model.fit(X_train, Y_train)

    y_pred_log = model.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(Y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
 
    print("======= BASIC METRICS (MEAN and MAX/MIN included - FEATURE ENGINEERED) =======")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    importances = pd.Series(
        model.feature_importances_,
        index=features_mean
    ).sort_values(ascending=True)

    print("\n Feature Importances: ")
    print(importances)
    
    #print(df['Foci'].describe())

#random_Forest_Regressor_meanANDmaxANDmin()
random_Forest_Regressor_meanANDmaxANDminV1()

def random_Forest_Regressor_Tuned_Model():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    df['rangeRH'] = df['maxRH'] - df['minRH']
    df['rangeSM'] = df['maxSM'] - df['minSM']
    df['rangeAT'] = df['maxAT'] - df['minAT']

    df['SM_AT'] = df['meanSM'] * df['meanAT']
    df['RH_AT'] = df['meanRH'] * df['meanAT']
    df['SM_RH'] = df['meanSM'] * df['meanRH']
    
    features_mean = ['meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT', 'maxLW', 'maxSM', 
                     'maxRH', 'maxAT', 'maxRF', 'minST', 'minSM', 'minRH', 'minAT',
                     'rangeRH', 'rangeSM', 'rangeAT', 'SM_AT', 'RH_AT', 'SM_RH']

    X = df[features_mean].dropna()
    Y = np.log1p(df.loc[X.index, 'Foci'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 8, 12, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.3, 0.5, 0.8]
    }

    search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist,
        n_iter=40,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, Y_train)

    best_model = search.best_estimator_

    y_pred_log = best_model.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(Y_test)

    print("\n ======= BEST PARAMETERS =======")
    print(search.best_params_)

    print("\n======= TUNED RANDOM FOREST PERFORMANCE =======")
    print(f"MAE : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R^2 : {r2_score(y_true, y_pred):.4f}")

#random_Forest_Regressor_Tuned_Model()