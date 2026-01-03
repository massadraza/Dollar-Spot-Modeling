import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
    
    #print(df['Foci'].describe())

random_Forest_Regressor_meanANDmaxANDminV1()