import torch
import pandas as pd
from design_forecast_lstm import load_trained_model, predict_next_month, generate_advice

# Define the paths for the trained model and data files
model_path = "predict_lstm.pt"  # Adjust the path as necessary
water_file_path = "./train_data/Monthly_Water_2021-2023.xlsx"
electricity_file_path = "./train_data/Monthly_electricity_2021-2023.xlsx"

# Load the trained model
model = load_trained_model(model_path)

# Load the water and electricity data
df_water = pd.read_excel(water_file_path, sheet_name="Sheet1")
df_electricity = pd.read_excel(electricity_file_path, sheet_name="Sheet1")

# Prepare the data for water usage
water_col = 'Total_Water(ML)'
series_water = df_water[water_col].dropna().values
months_water = df_water.loc[df_water[water_col].dropna().index, 'Month'].values

# Prepare the data for electricity usage
elec_col = 'Total Electricity Consumption (kWh)'
series_elec = df_electricity[elec_col].dropna().values
months_elec = df_electricity.loc[df_electricity[elec_col].dropna().index, 'Month'].values

# Predict the next month's water and electricity usage
next_water = predict_next_month(series_water, months_water, model_path)
next_elec = predict_next_month(series_elec, months_elec, model_path)

# Generate advice based on the predictions
curr_water = series_water[-1]
curr_elec = series_elec[-1]

water_diff = next_water - curr_water
elec_diff = next_elec - curr_elec

water_pct = water_diff / curr_water * 100
elec_pct = elec_diff / curr_elec * 100

trend_water = "上升" if water_diff > 1e-3 else "下降" if water_diff < -1e-3 else "持平"
trend_elec = "上升" if elec_diff > 1e-3 else "下降" if elec_diff < -1e-3 else "持平"

# Prepare the prompt for LLM advice generation
prompt = (
    f"你是一位節能與節水顧問，請依據以下數據條列3點繁體中文建議。\n"
    f"- 本月用水 {curr_water:.2f} ML，預估下月 {next_water:.2f} ML，{trend_water} {water_pct:+.1f}%\n"
    f"- 本月用電 {curr_elec:.2f} kWh，預估下月 {next_elec:.2f} kWh，{trend_elec} {elec_pct:+.1f}%\n"
    f"可以從一些日常習慣與常見的電器使用方式來建議。"
)

advice = generate_advice(prompt)

# Output the results
print("============ 用水 / 用電 預測概覽 ============")
print(f"🔹 本月用水量：{curr_water:,.2f} ML")
print(f"🔹 下月預估用水量：{next_water:,.2f} ML（{trend_water} {water_pct:+.1f}%）")
print(f"🔹 本月用電量：{curr_elec:,.2f} kWh")
print(f"🔹 下月預估用電量：{next_elec:,.2f} kWh（{trend_elec} {elec_pct:+.1f}%）")

# Print the LLM advice
print("\n============ 建議 ============")
print(advice)
print("================================")