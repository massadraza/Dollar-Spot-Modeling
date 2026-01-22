import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def random_Forest_Regressor_meanANDmaxANDminV1_MoreFeatures():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    df['rangeRH'] = df['maxRH'] - df['minRH']
    df['rangeSM'] = df['maxSM'] - df['minSM']
    df['rangeAT'] = df['maxAT'] - df['minAT']

    df['SM_AT'] = df['meanSM'] * df['meanAT']
    df['RH_AT'] = df['meanRH'] * df['meanAT']
    df['SM_RH'] = df['meanSM'] * df['meanRH']
    
    features_mean = ['meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT', 'maxLW', 'maxSM', 
                     'maxRH', 'maxAT', 'maxRF', 'minST', 'minSM', 'minRH', 'minAT',
                     'rangeRH', 'rangeSM', 'rangeAT', 'SM_AT', 'RH_AT', 'SM_RH','DayOfYear', 'Rating']

    X = df[features_mean].dropna()
    Y = np.log1p(df.loc[X.index, 'Foci'])
    
    X = X.drop(columns=['maxSM', 'maxRH', 'maxRF', 'minSM', 'minAT'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        min_samples_leaf=7,
        n_jobs=-1
     #   max_depth=10,
     #max_features="sqrt"
    )

    model.fit(X_train, Y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(Y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
 
    print("======= BASIC METRICS (MEAN and MAX/MIN included - FEATURE ENGINEERED) - more features =======")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")



def grid_search_random_forest():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    df['rangeRH'] = df['maxRH'] - df['minRH']
    df['rangeSM'] = df['maxSM'] - df['minSM']
    df['rangeAT'] = df['maxAT'] - df['minAT']

    df['SM_AT'] = df['meanSM'] * df['meanAT']
    df['RH_AT'] = df['meanRH'] * df['meanAT']
    df['SM_RH'] = df['meanSM'] * df['meanRH']

    features_mean = ['meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT', 'maxLW', 'maxSM',
                     'maxRH', 'maxAT', 'maxRF', 'minST', 'minSM', 'minRH', 'minAT',
                     'rangeRH', 'rangeSM', 'rangeAT', 'SM_AT', 'RH_AT', 'SM_RH','DayOfYear', 'Rating']

    X = df[features_mean].dropna()
    Y = np.log1p(df.loc[X.index, 'Foci'])

    X = X.drop(columns=['maxSM', 'maxRH', 'maxRF', 'minSM', 'minAT'])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 4, 7],
        'max_features': ['sqrt', 'log2', None]
    }

    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    print("Running GridSearchCV...")
    grid_search.fit(X_train, Y_train)

    print("\n======= GRID SEARCH CV RESULTS =======")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (neg MSE): {grid_search.best_score_:.4f}")

    model = grid_search.best_estimator_

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(Y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n======= TEST SET METRICS (Best Model) =======")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")


random_Forest_Regressor_meanANDmaxANDminV1_MoreFeatures()
#grid_search_random_forest()