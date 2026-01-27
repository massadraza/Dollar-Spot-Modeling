import pandas as pd

df = pd.read_csv('2023Chemistry.csv')

df = df[df['trt'].str.contains('Non', na=False)]

df.to_csv("2023Chemistry_V2.csv")