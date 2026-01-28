import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DOLLAR SPOT FOCI PREDICTION - MODEL TRAINING")
print("="*70)

# Load data
df = pd.read_csv('combined_foci_df.csv')
print(f"\nDataset: {df.shape[0]} samples, {df.shape[1]} columns")

# Log-transform target
df['Foci_log'] = np.log1p(df['Foci'])

# Feature engineering
df['AT_range'] = df['maxAT'] - df['minAT']
df['RH_range'] = df['maxRH'] - df['minRH']
df['temp_humidity'] = df['meanAT'] * df['meanRH']
df['dewpoint_diff'] = df['meanAT'] - df['meanDP']
df['moisture_temp'] = df['meanSM'] * df['meanST']
df['leafwet_humidity'] = df['meanLW'] * df['meanRH']
df['favorable_conditions'] = ((df['meanAT'] > 15) & (df['meanRH'] > 80) & (df['meanLW'] > 30)).astype(int)

# Prepare features
feature_cols = [c for c in df.columns if c not in ['Date', 'Foci', 'Foci_log']]
X = df[feature_cols]
y_log = df['Foci_log']
y_original = df['Foci']

print(f"Features ({len(feature_cols)}): {feature_cols}")

# Stratified split
y_bins = pd.cut(y_log, bins=5, labels=False)
X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
    X, y_log, y_original, test_size=0.2, random_state=42, stratify=y_bins
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# DEFINE ALL MODELS
# ============================================================
models = {
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=42
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=5,
        max_features='sqrt', random_state=42, n_jobs=-1
    ),
    'Ridge': Ridge(alpha=10.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
}

# ============================================================
# TRAIN AND EVALUATE ALL MODELS
# ============================================================
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...", end=" ")

    # Linear models and SVR need scaled data
    needs_scaling = name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR']

    if needs_scaling:
        model.fit(X_train_scaled, y_train)
        y_pred_log = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    # Convert predictions back
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)

    # Metrics
    r2 = r2_score(y_test_orig, y_pred)
    r2_log = r2_score(y_test, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    mae = mean_absolute_error(y_test_orig, y_pred)

    results[name] = {
        'R2': r2, 'R2_log': r2_log, 'CV_R2': cv_scores.mean(),
        'CV_std': cv_scores.std(), 'RMSE': rmse, 'MAE': mae,
        'needs_scaling': needs_scaling
    }
    trained_models[name] = model

    print(f"R²={r2_log:.3f} (log), CV={cv_scores.mean():.3f}")

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "="*70)
print("MODEL COMPARISON (sorted by log-scale R²)")
print("="*70)
print(f"{'Model':<18} {'R² (log)':<10} {'CV R²':<12} {'R² (orig)':<10} {'MAE':<8}")
print("-"*70)

sorted_models = sorted(results.items(), key=lambda x: x[1]['R2_log'], reverse=True)
for name, r in sorted_models:
    print(f"{name:<18} {r['R2_log']:<10.4f} {r['CV_R2']:.4f}±{r['CV_std']:.3f}  {r['R2']:<10.4f} {r['MAE']:<8.2f}")

# ============================================================
# SAVE ALL MODELS
# ============================================================
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

# Save scaler
joblib.dump(scaler, 'scaler.joblib')
print("Saved: scaler.joblib")

# Save feature list
joblib.dump(feature_cols, 'feature_columns.joblib')
print("Saved: feature_columns.joblib")

# Save each model
for name, model in trained_models.items():
    filename = f'model_{name}.joblib'
    joblib.dump(model, filename)
    print(f"Saved: {filename}")

# Save results summary
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')
print("Saved: model_results.csv")

# ============================================================
# BEST MODEL
# ============================================================
best_name = sorted_models[0][0]
best_r2 = sorted_models[0][1]['R2_log']

print("\n" + "="*70)
print(f"BEST MODEL: {best_name}")
print(f"  Log-scale R²: {best_r2:.4f}")
print(f"  CV R²: {results[best_name]['CV_R2']:.4f}")
print("="*70)

# ============================================================
# FEATURE IMPORTANCE (for tree models)
# ============================================================
print("\nFeature Importance (Gradient Boosting):")
print("-"*50)
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': trained_models['GradientBoosting'].feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.head(10).iterrows():
    bar = "█" * int(row['Importance'] * 80)
    print(f"{row['Feature']:20} {row['Importance']:.4f} {bar}")

importance.to_csv('feature_importance.csv', index=False)
print("\nSaved: feature_importance.csv")
