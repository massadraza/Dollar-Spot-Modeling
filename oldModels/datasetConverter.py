import numpy as np
import pandas as pd

def shoulderSeasonsOnly():

    df = pd.read_csv("Smith-Kerns Dataset 2026.csv")

    numeric_cols = [
        'Foci', 'meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT', 'meanRF', 
        'maxLW', 'maxST', 'maxSM', 'maxRH', 'maxAT', 'maxRF', 'minST', 'minSM',
        'minRH', 'minAT'
    ]

    for col in numeric_cols:
        df[col] = df[col].replace('.', np.nan).astype(float)
    
    # Drop columns that are missing Foci values
    df_cleaned = df.dropna(subset=['Foci']).copy()

    # Correctly formats date for model usage
    df_cleaned['Date_str'] = df_cleaned['Date'].astype(int).astype(str).str.zfill(6)

    # Converted into a usable date time object
    df_cleaned['Date_dt'] = pd.to_datetime(df_cleaned['Date_str'], format='%m%d%y')

    # Returns a column of ordinal position (# of days from Jan 1 etc.)
    df_cleaned['DayOfYear'] = df_cleaned['Date_dt'].dt.dayofyear

    SPRING_SHOULDER_END = 165 # Up to and including June 14th
    FALL_SHOULDER_START = 259 # From September 16th (including the 16th)

    spring_shoulder = df_cleaned['DayOfYear'] <= SPRING_SHOULDER_END
    fall_shoulder = df_cleaned['DayOfYear'] >= FALL_SHOULDER_START

    df_shoulder_season = df_cleaned[spring_shoulder | fall_shoulder].copy()
    df_shoulder_season = df_shoulder_season.drop(['Date_dt'], axis=1)    
    df_shoulder_season = df_shoulder_season.drop(['Date_str'], axis=1)
    df_shoulder_season = df_shoulder_season[df_shoulder_season['Treatment'].str.contains('CHECK', na=False)]

    df_shoulder_season.to_csv('Shoulder_Season_Data.csv', index=False)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print(df_shoulder_season)

    return df_shoulder_season

def counter():
    df = pd.read_csv("Smith-Kerns Dataset 2026.csv")

    print((df == ".").sum())



#shoulderSeasonsOnly()

# 846 Observations

counter()
