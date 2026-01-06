import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def random_Forest_Regressor_meanONLY():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    features_mean = ['meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT']

    X = df[features_mean].dropna()
    Y = df.loc[X.index, 'Foci']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # ----> Use all available CPU cores
    )

    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(Y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
    r2 = r2_score(Y_test, y_pred)
    
    print("======= BASIC METRICS (MEAN ONLY) =======")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

#random_Forest_Regressor_meanONLY()

"""
Using a mean only approach is not good because the model cannot account 
for mins and maxes which will hurt the RMSE which penalizes LARGE errors

"""

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
   
    """
    importances = model.feature_importances_

    feature_importances_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()

    """
"""
Accuracy has somewhat improved but more can still be done


"""

random_Forest_Regressor_meanANDmaxANDmin()


# Done STEP 1