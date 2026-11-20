import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from xgboost import XGBRegressor

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
df1 = pd.read_excel("data/2023.xlsx")
df2 = pd.read_csv("data/2022.csv")

# Drop empty NaN rows caused by Excel spacing
df1.dropna(inplace=True)
df2.dropna(inplace=True)
df1.reset_index(inplace=True, drop=True)
df2.reset_index(inplace=True, drop=True)

# Merge 2022 stock into 2023 dataframe
df1['STOCK-2022'] = df2['STOCK']
df1.rename(columns={'STOCK': 'STOCK-2023'}, inplace=True)

# Convert DATE to datetime and set as index
df1['DATE'] = pd.to_datetime(df1['DATE'])
df1.set_index('DATE', inplace=True)

# Feature Engineering: Previous day's stock (lag feature)
df1['lag_1-STOCK-2023'] = df1['STOCK-2023'].shift(1)
df1.dropna(inplace=True)

# Move target column ('STOCK-2023') to the rightmost position
col_to_move = df1.pop('STOCK-2023')
df1['STOCK-2023'] = col_to_move

# ==========================================
# 2. EXPLORATORY VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))
plt.plot(df1.index, df1['STOCK-2023'], label='Stock 2023')
plt.title('Daily Inventory Variation (2023)')
plt.xlabel('Date')
plt.ylabel('Stock Level')
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 3. TRAIN / TEST SPLIT & SCALING (80/20)
# ==========================================
train_size = 0.8
split_index = int(len(df1) * train_size)
train_df, test_df = df1[:split_index], df1[split_index:]

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_df)
test_data_scaled = scaler.transform(test_df)

# Features (X): PRODUCTION, VENTE, STOCK-2022, lag_1-STOCK-2023
# Target (y): STOCK-2023
X_train, y_train = train_data_scaled[:, :-1], train_data_scaled[:, -1]
X_test, y_test = test_data_scaled[:, :-1], test_data_scaled[:, -1]

# ==========================================
# 4. EVALUATION & DESCALING FUNCTIONS
# ==========================================
def calculate_metrics(model_name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    print(f"--- {model_name} ---")
    print(f"MSE:  {mse:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE:  {mae:.5f}")
    print(f"MAPE: {mape:.5f}\n")

def descale(predictions, y_true):
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
    scaled_pred = descaler.inverse_transform(predictions.reshape(-1, 1))
    scaled_y = descaler.inverse_transform(y_true.reshape(-1, 1))
    return scaled_pred, scaled_y

def plot_actual_vs_prediction(pred, title):
    plt.figure(figsize=(10, 6))
    pred_plot, y_plot = descale(pred, y_test)
    plt.plot(test_df.index, y_plot, label='Actual')
    plt.plot(test_df.index, pred_plot, label='Predicted', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Stock')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. MODEL 1: XGBOOST & GRIDSEARCHCV
# ==========================================
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
predictions_xgb = xgb_model.predict(X_test)
calculate_metrics("XGBoost (Baseline)", y_test, predictions_xgb)

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_xgb = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid_xgb,
    cv=3,
    scoring='neg_mean_squared_error'
)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best XGBoost Parameters:", grid_xgb.best_params_)

predictions_grid_xgb = best_xgb.predict(X_test)
calculate_metrics("XGBoost (GridSearchCV)", y_test, predictions_grid_xgb)
plot_actual_vs_prediction(predictions_grid_xgb, "Actual vs Predicted Stock (Tuned XGBoost)")

# ==========================================
# 6. MODEL 2: NEURAL NETWORK (MLP) & GRIDSEARCHCV
# ==========================================
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='sgd',
    max_iter=1000,
    random_state=42
)
mlp_model.fit(X_train, y_train)
predictions_mlp = mlp_model.predict(X_test)
calculate_metrics("Neural Network (Baseline)", y_test, predictions_mlp)

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [500, 1000, 1500]
}
grid_mlp = GridSearchCV(
    estimator=MLPRegressor(random_state=42),
    param_grid=param_grid_mlp,
    cv=3,
    scoring='neg_mean_squared_error'
)
grid_mlp.fit(X_train, y_train)
best_mlp = grid_mlp.best_estimator_
print("Best MLP Parameters:", grid_mlp.best_params_)

predictions_grid_mlp = best_mlp.predict(X_test)
calculate_metrics("Neural Network (GridSearchCV)", y_test, predictions_grid_mlp)
plot_actual_vs_prediction(predictions_grid_mlp, "Actual vs Predicted Stock (Tuned MLP)")

# ==========================================
# 7. EXPORT PREDICTIONS TO EXCEL
# ==========================================
pred_final, actual_final = descale(predictions_grid_xgb, y_test)
results_df = pd.DataFrame({
    'Date': test_df.index.strftime('%Y-%m-%d'),
    'Actual': actual_final.flatten(),
    'Predicted': pred_final.flatten()
}).set_index('Date')

results_df.to_excel("predictions.xlsx", index=True)
print("Saved predictions.xlsx successfully!")