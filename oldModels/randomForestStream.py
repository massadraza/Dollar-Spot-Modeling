import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap


st.title("Random Forest Model Analysis - Shoulder Season Data")

@st.cache_data
def load_data():
    df = pd.read_csv("Shoulder_Season_Data.csv")
    # Feature engineering
    df['rangeRH'] = df['maxRH'] - df['minRH']
    df['rangeSM'] = df['maxSM'] - df['minSM']
    df['rangeAT'] = df['maxAT'] - df['minAT']
    df['SM_AT'] = df['meanSM'] * df['meanAT']
    df['RH_AT'] = df['meanRH'] * df['meanAT']
    df['SM_RH'] = df['meanSM'] * df['meanRH']
    return df

df = load_data()
st.subheader("Dataset Sample")
st.dataframe(df.head())

features_mean = ['meanLW', 'meanST', 'meanSM', 'meanRH', 'meanAT', 'maxLW', 'maxSM', 
                 'maxRH', 'maxAT', 'maxRF', 'minST', 'minSM', 'minRH', 'minAT',
                 'rangeRH', 'rangeSM', 'rangeAT', 'SM_AT', 'RH_AT', 'SM_RH', 'Obs', 'Rep', 'Rating', 'Date']

X = df[features_mean].dropna()
Y = np.log1p(df.loc[X.index, 'Foci'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

st.sidebar.subheader("Random Forest Hyperparameters")
n_estimators = st.sidebar.slider("n_estimators", min_value=50, max_value=500, value=150, step=50)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", min_value=1, max_value=20, value=7, step=1)

model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=42,
    min_samples_leaf=min_samples_leaf,
    n_jobs=-1
)
model.fit(X_train, Y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(Y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

st.subheader("Model Metrics")
st.write(f"**MAE:** {mae:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**R²:** {r2:.4f}")

importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)


st.subheader("Feature Importance")
st.dataframe(feat_imp_df)

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, ax=ax)
ax.set_title("Feature Importances")
st.pyplot(fig)

st.subheader("Residual Analysis")
fig2, ax2 = plt.subplots()
ax2.scatter(y_true, y_pred, alpha=0.5)
ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
ax2.set_xlabel("True Foci")
ax2.set_ylabel("Predicted Foci")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
residuals = y_true - y_pred
ax3.hist(residuals, bins=50)
ax3.set_title("Residual Distribution")
st.pyplot(fig3)

st.subheader("SHAP Feature Effects")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

fig4 = plt.figure()
shap.summary_plot(shap_values, X_train, show=False)
st.pyplot(fig4)
