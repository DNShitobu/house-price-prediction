# 🏠 House Price Prediction

Regression ML project predicting California median house prices using scikit-learn with feature engineering and multiple model comparison.

## Overview
| Detail | Value |
|--------|-------|
| Type | Regression |
| Dataset | California Housing (sklearn, 20,640 samples) |
| Framework | scikit-learn |
| Models | Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting |

## Getting Started
```bash
git clone https://github.com/Dnshitobu/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
python house_price_prediction.py
```

## What It Does
1. Loads California Housing dataset and runs EDA
2. Adds feature engineering (rooms per household, bedroom ratio, population density)
3. Trains 6 regression models
4. Evaluates with MAE, RMSE, and R² score
5. Plots residuals and feature importances

## Results
| Model | R² | RMSE |
|-------|:---:|:---:|
| Linear Regression | ~0.60 | ~0.72 |
| **Random Forest** | **~0.80** | **~0.51** |
| Gradient Boosting | ~0.78 | ~0.53 |

## Concepts Covered
Regression metrics (MAE/RMSE/R²) · Feature engineering · Regularization · Ensemble methods
