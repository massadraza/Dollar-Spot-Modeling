import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

MAPPED_FEATURES = [
    'maxAT', 'meanAT', 'minAT', 
    'meanRH', 'maxRH', 'minRH',
    'meanDP', 'meanLW', 'meanSM',
    'meanST', 'meanRF'
]

TARGET = 'Foci'

# Load the datasets
new_foci_df = pd.read_csv('New-Foci-Dataset.csv')
shoulder_df = pd.read_csv('Shoulder_Season_Data_Formula.csv')

columns_to_keep = ['Date'] + MAPPED_FEATURES + [TARGET]

new_foci_subset = new_foci_df[columns_to_keep].copy()
shoulder_subset = shoulder_df[columns_to_keep].copy()

combined_df = pd.concat([new_foci_subset, shoulder_subset], ignore_index=True)

combined_df = combined_df.dropna()

combined_df.to_csv('combined_foci_df.csv', index=False)
