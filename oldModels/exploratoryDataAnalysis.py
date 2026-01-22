import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Shoulder_Season_Data.csv")

# Foci Histogram of Distribution
plt.figure()
plt.hist(df["Foci"], bins=30)
plt.xlabel("Foci")
plt.ylabel("Frequency")
plt.title("Distribution of Foci")
plt.show()

# Foci vs Day of Year
plt.figure()
plt.scatter(df["DayOfYear"], df["Foci"])
plt.xlabel("Day of Year")
plt.ylabel("Foci")
plt.title("Foci vs Day of Year")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()