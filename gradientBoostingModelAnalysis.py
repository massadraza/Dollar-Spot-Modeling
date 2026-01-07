import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

def gradient_boosting_model_encoded_FEATURE_ENGINEERED():
    df = pd.read_csv("Shoulder_Season_Data.csv")

    feature_cols = [
        "Rating", "Date", "Year", "Season", "Obs", "Rep",
        "meanLW", "meanST", "meanSM", "meanRH", "meanAT", "meanRF",
        "maxLW", "maxST", "maxSM", "maxRH", "maxAT", "maxRF",
        "minST", "minSM", "minRH", "minAT"
    ]

    X = df[feature_cols].copy()
    y = df["Foci"]
    y_log = np.log1p(y)

    X['rangeRH'] = X['maxRH'] - X['minRH']
    X['rangeSM'] = X['maxSM'] - X['minSM']
    X['rangeAT'] = X['maxAT'] - X['minAT']

    X['SM_AT'] = X['meanSM'] * X['meanAT']
    X['RH_AT'] = X['meanRH'] * X['meanAT']
    X['SM_RH'] = X['meanSM'] * X['meanRH']

    categorical_cols = ["Season"]

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough"
    )
    
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
   
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("=========== GRADIENT BOOSTING METRICS (FEATURE ENGINEERED) ===========")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    top_3_idx, top_3_impacts, all_impacts = top_k_most_impactful_gb_trees(
        pipeline=model,
        X=X_test,
        k = 3
    )

    print("Top 3 most impactful trees:", top_3_idx)
    print("Impact values:", top_3_impacts)

    visualize_gb_trees(
        pipeline=model,
        tree_indices=top_3_idx,
        max_depth=3
    )



def top_k_most_impactful_gb_trees(
        pipeline, X, k=3
):
    preprocess = pipeline.named_steps["preprocess"]
    gbr = pipeline.named_steps["model"]

    X_trans = preprocess.transform(X)
    stage_preds = list(gbr.staged_predict(X_trans))
    tree_impacts = []
    prev = np.zeros_like(stage_preds[0])

    for preds in stage_preds:
        delta = preds - prev
        tree_impacts.append(np.mean(np.abs(delta)))
        prev = preds

    tree_impacts = np.array(tree_impacts)

    top_indices = np.argsort(tree_impacts)[-k:][::-1]
    top_impacts = tree_impacts[top_indices]

    return top_indices, top_impacts, tree_impacts

def visualize_gb_trees(
        pipeline,
        tree_indices, 
        max_depth=None,
        figsize=(18, 8)
):
    gbr = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()

    for idx in tree_indices:
        plt.figure(figsize=figsize)
        plot_tree(
            gbr.estimators_[idx, 0],
            feature_names=feature_names,
            filled=True,
            rounded=True,
            max_depth=max_depth,
            fontsize=10
        )
        plt.title(f"Gradient Boosting Tree #{idx}")
        plt.show()


gradient_boosting_model_encoded_FEATURE_ENGINEERED()