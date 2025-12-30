import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def shoulderSeasonsOnly():

    df = pd.read_csv("Smith-Kerns Dataset 2026.xls")

    numeric_cols = [
        'Foci', 'meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT', 'meanRF', 
        'maxLW', 'maxST', 'maxSM', 'maxRH', 'maxAT', 'maxRF', 'minST', 'minSM',
        'minRH', 'minAT'
    ]

    for col in numeric_cols:
        df[col] = df[col].replace('.', np.nan).astype(float)
    
    # Drop columns that are missing Foci values
    df_cleaned = df.dropna(subset=['Foci']).copy()

    SPRING_SHOULDER_END = 165 # Up to and including June 14th
    FALL_SHOULDER_START = 259 # From September 16th (including the 16th)

    spring_shoulder = df_cleaned['DayOfYear'] <= SPRING_SHOULDER_END
    fall_shoulder = df_cleaned['DayOfYear'] >= FALL_SHOULDER_START