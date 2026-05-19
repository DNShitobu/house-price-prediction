"""
House Price Prediction
======================
Predicts house prices using regression models on the California Housing dataset.
Covers EDA, feature engineering, multiple models, residual analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
target_col = "MedHouseVal"

print("=" * 60)
print("HOUSE PRICE PREDICTION")
print("=" * 60)
print(f"Shape: {df.shape}")
print(df.describe().round(2))

# EDA
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
df[target_col].hist(bins=50, color="steelblue", edgecolor="white")
plt.title("Target Distribution")
plt.subplot(1, 2, 2)
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation")
plt.tight_layout()
plt.savefig("eda.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature engineering
df["rooms_per_household"] = df["AveRooms"] / df["HouseAge"].clip(lower=1)
df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"].clip(lower=1)
df["population_per_household"] = df["Population"] / df["AveOccup"].clip(lower=1)

feature_cols = [c for c in df.columns if c != target_col]
X, y = df[feature_cols], df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    results[name] = {"MAE": mean_absolute_error(y_test, preds),
                     "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                     "R2": r2_score(y_test, preds), "preds": preds}
    print(f"{name}: R2={results[name]['R2']:.4f} RMSE={results[name]['RMSE']:.4f}")

best = max(results, key=lambda k: results[k]["R2"])
print(f"\nBest model: {best} (R2={results[best]['R2']:.4f})")

rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="mediumseagreen")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Done!")
