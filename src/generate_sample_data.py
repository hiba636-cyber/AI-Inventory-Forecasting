import pandas as pd
import numpy as np

np.random.seed(42)

# Generate 2022 sample data (matching your report's structure & scale)
dates_2022 = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
prod_2022 = np.random.randint(420000, 480000, size=len(dates_2022)).astype(float)
vente_2022 = prod_2022 - np.random.randint(10000, 15000, size=len(dates_2022))
stock_2022 = np.random.randint(29000, 35000, size=len(dates_2022)).astype(float)

df_2022 = pd.DataFrame({
    "DATE": dates_2022.strftime("%-m/%-d/%Y"),
    "PRODUCTION": prod_2022,
    "VENTE": vente_2022,
    "STOCK": stock_2022
})
df_2022.to_csv("data/2022.csv", index=False)

# Generate 2023 sample data (matching your report's structure & scale)
dates_2023 = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
prod_2023 = np.random.randint(530000, 605000, size=len(dates_2023)).astype(float)
vente_2023 = prod_2023 - np.random.randint(12000, 14000, size=len(dates_2023))
stock_2023 = np.random.randint(30500, 34500, size=len(dates_2023)).astype(float)

df_2023 = pd.DataFrame({
    "DATE": dates_2023,
    "PRODUCTION": prod_2023,
    "VENTE": vente_2023,
    "STOCK": stock_2023
})
df_2023.to_excel("data/2023.xlsx", index=False)

print("Created data/2022.csv and data/2023.xlsx successfully!")