# AI-Driven Inventory & Stock Level Forecasting

## Project Overview
This project applies Machine Learning and Deep Learning techniques to optimize inventory management and daily stock forecasting for a high-volume fast-moving consumer goods (FMCG) beverage distribution company[cite: 4, 70]. Using historical production, sales, and lagged stock data, the models predict daily inventory levels to minimize stockouts and overstocking costs[cite: 4, 70, 75, 77].

*(Note: Proprietary company identifiers have been anonymized to ensure data confidentiality.)*

## Repository Structure
- `src/stock_forecasting.py`: Complete data cleaning, feature engineering, model training, hyperparameter tuning, and visualization pipeline[cite: 72, 74, 76, 78, 79, 80].
- `requirements.txt`: Required Python dependencies.

## Methodology
1. **Data Cleaning & Alignment:** Removing empty rows (`NaN`), aligning 2022 and 2023 datasets, and engineering a 1-day lagged stock feature (`lag_1-STOCK-2023`)[cite: 72, 73, 75].
2. **Preprocessing:** Chronological 80/20 train-test split and `MinMaxScaler` normalization[cite: 77].
3. **Models Evaluated:** **XGBoost Regressor** and **Multi-Layer Perceptron (`MLPRegressor`)**, both optimized using `GridSearchCV`[cite: 78, 80, 82, 83].

## Model Performance Comparison
| Model | MSE | RMSE | MAE | MAPE |
| :--- | :---: | :---: | :---: | :---: |
| Neural Network (Baseline)[cite: 86] | 0.00179[cite: 82, 86] | 0.04235[cite: 82, 86] | 0.03398[cite: 82, 86] | 7.72%[cite: 82, 86] |
| Neural Network (GridSearch)[cite: 86] | 0.00173[cite: 83, 86] | 0.04164[cite: 83, 86] | 0.03272[cite: 83, 86] | 7.74%[cite: 83, 86] |
| XGBoost (Baseline)[cite: 86] | 0.00233[cite: 78, 86] | 0.04828[cite: 78, 86] | 0.03909[cite: 78, 86] | 9.20%[cite: 78, 86] |
| **XGBoost (GridSearch)**[cite: 86] | **0.00135**[cite: 80, 86] | **0.03669**[cite: 80, 86] | **0.02981**[cite: 80, 86] | **6.94%**[cite: 80, 86] |