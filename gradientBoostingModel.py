import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv('combined_foci_df_fixed.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Aggregate to daily means (reduces noise from multiple plots)
df_daily = df.groupby('Date').mean().reset_index()
df_daily = df_daily.sort_values('Date').reset_index(drop=True)

print(f"Original: {len(df)} rows --> Daily aggregated: {len(df_daily)} days")

# ============================================================
# 5-DAY MOVING AVERAGE (biologically motivated lag)
# Pathogens need ~5 days of conducive conditions before symptoms
# ============================================================
weather_cols = ['meanAT', 'meanRH', 'meanLW', 'meanDP', 'meanSM', 'meanST', 'meanRF']

for col in weather_cols:
    # 5-day moving average, shifted by 1 to avoid data leakage
    df_daily[f'{col}_5d'] = df_daily[col].rolling(5).mean().shift(1)

# Interaction features on 5-day averages
df_daily['temp_humidity_5d'] = df_daily['meanAT_5d'] * df_daily['meanRH_5d']
df_daily['leafwet_humidity_5d'] = df_daily['meanLW_5d'] * df_daily['meanRH_5d']
df_daily['dewpoint_diff_5d'] = df_daily['meanAT_5d'] - df_daily['meanDP_5d']

# Previous day's foci (disease momentum)
df_daily['Foci_lag1'] = df_daily['Foci'].shift(1)

# Drop NaN from rolling/lag features
df_daily = df_daily.dropna()

# Filter extreme outbreaks (top 3%)
df_daily = df_daily[df_daily['Foci'] <= 100]

print(f"After filtering (Foci <= 100): {len(df_daily)} days")

# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================
feature_cols = (
    [f'{c}_5d' for c in weather_cols] +  # 5-day weather averages
    ['temp_humidity_5d', 'leafwet_humidity_5d', 'dewpoint_diff_5d']  +  # Interactions
     ['Foci_lag1']  # Previous day's foci
)

X = df_daily[feature_cols]
y = df_daily['Foci']

print(f"Features: {len(feature_cols)}")
print(f"  5-day weather averages: {len(weather_cols)}")
print(f"  Interaction features: 3")
print(f"  Foci lag: 1")

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
print("RESULTS (5-day moving average model)")
print("="*50)
print(f"R² (original scale): {r2:.4f}")
print(f"CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")



# ============================================================
# FEATURE IMPORTANCE
# ============================================================
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print("-"*50)
for _, row in importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"{row['Feature']:22} {row['Importance']:.4f} {bar}")