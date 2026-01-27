import pandas as pd

df = pd.read_csv('NewFoci_v15.csv')

df['Foci'] = df['Damage'] / 176.7

df.to_csv("NewConversion.csv", index=False)