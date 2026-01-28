import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('combined_foci_df_fixed.csv')

# 1. Basic Info
print("="*50)
print("BASIC INFO")
print("="*50)
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# 2. Summary Statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(df.describe())

# 3. Target Variable (Foci) Distribution
print("\n" + "="*50)
print("FOCI DISTRIBUTION")
print("="*50)
print(f"Min: {df['Foci'].min():.4f}")
print(f"Max: {df['Foci'].max():.4f}")
print(f"Mean: {df['Foci'].mean():.4f}")
print(f"Median: {df['Foci'].median():.4f}")
print(f"Std: {df['Foci'].std():.4f}")
print(f"Skewness: {df['Foci'].skew():.4f}")

# 4. Correlation Matrix
print("\n" + "="*50)
print("CORRELATION WITH FOCI")
print("="*50)
correlations = df.select_dtypes(include=[np.number]).corr()['Foci'].sort_values(ascending=False)
print(correlations)

# 5. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Foci histogram
axes[0, 0].hist(df['Foci'], bins=50, edgecolor='black')
axes[0, 0].set_xlabel('Foci')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Foci Distribution')

# Foci log-transformed histogram
axes[0, 1].hist(np.log1p(df['Foci']), bins=50, edgecolor='black', color='orange')
axes[0, 1].set_xlabel('Log(Foci + 1)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Log-Transformed Foci Distribution')

# Boxplot of features
df_features = df[['meanAT', 'meanRH', 'meanDP', 'meanLW', 'meanSM', 'meanST']]
axes[1, 0].boxplot(df_features.values, labels=df_features.columns)
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Feature Boxplots')
axes[1, 0].tick_params(axis='x', rotation=45)

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            ax=axes[1, 1], annot_kws={'size': 8})
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150)
plt.show()

print("\nPlots saved to: eda_plots.png")
