#############
# BASELINE TESTING MODEL
#############

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('combined_foci_df_fixed.csv')

print(f"Dataset: {len(df)} rows")

feature_cols = ['maxAT', 'meanAT', 'minAT', 'meanRH', 'maxRH', 'minRH',
                'meanDP', 'meanLW', 'meanSM', 'meanST', 'meanRF']

X = df[feature_cols]
y = df['Foci']

print(f"Features: {len(feature_cols)}")
print(f"  {feature_cols}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")


model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)


y_pred = np.maximum(model.predict(X_test), 0)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("\n" + "="*50)
print("BASELINE RESULTS (original features, no aggregation)")
print("="*50)
print(f"R² (original scale): {r2:.4f}")
print(f"CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)


print("\nFeature Importance:")
print("-"*50)
for _, row in importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"{row['Feature']:12} {row['Importance']:.4f} {bar}")

"""
==================================================
BASELINE RESULTS (original features, no aggregation)
==================================================
R² (original scale): 0.0307
CV R² (5-fold): -160.0496 ± 313.8870
RMSE: 271.63
MAE: 73.27

"""
