# AI-Driven Inventory & Stock Level Forecasting

## Project Overview
This project applies Machine Learning and Deep Learning techniques to optimize inventory management and daily stock forecasting for a high-volume fast-moving consumer goods (FMCG) beverage distribution company. Using historical production, sales, and lagged stock data, the models predict daily inventory levels to minimize stockouts and overstocking costs.

*(Note: Proprietary company identifiers have been anonymized to ensure data confidentiality.)*

## Repository Structure
- `src/stock_forecasting.py`: Complete data cleaning, feature engineering, model training, hyperparameter tuning, and visualization pipeline.
- `requirements.txt`: Required Python dependencies.

## Methodology
1. **Data Cleaning & Alignment:** Removing empty rows (`NaN`), aligning 2022 and 2023 datasets, and engineering a 1-day lagged stock feature (`lag_1-STOCK-2023`).
2. **Preprocessing:** Chronological 80/20 train-test split and `MinMaxScaler` normalization.
3. **Models Evaluated:** **XGBoost Regressor** and **Multi-Layer Perceptron (`MLPRegressor`)**, both optimized using `GridSearchCV`.

## Model Performance Comparison
| Model | MSE | RMSE | MAE | MAPE |
| :--- | :---: | :---: | :---: | :---: |
| Neural Network (Baseline) | 0.00179 | 0.04235 | 0.03398 | 7.72% |
| Neural Network (GridSearch) | 0.00173 | 0.04164 | 0.03272 | 7.74% |
| XGBoost (Baseline) | 0.00233 | 0.04828 | 0.03909 | 9.20% |
| **XGBoost (GridSearch)** | **0.00135** | **0.03669** | **0.02981** | **6.94%** |