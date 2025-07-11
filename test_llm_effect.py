# -*- coding: utf-8 -*-
"""
測試 LLM 產生建議效果（不進行預測，只產生建議）
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 你可以根據需求修改這些數值
curr_water = 123.45  # 本月用水量 (ML)
next_water = 120.00  # 下月預估用水量 (ML)
trend_water = "下降"
water_pct = -2.8

curr_elec = 6789.0   # 本月用電量 (kWh)
next_elec = 7000.0   # 下月預估用電量 (kWh)
trend_elec = "上升"
elec_pct = +3.1

# =========  LLM載入 =========
model_name = "Qwen/Qwen1.5-0.5B"
# model_name = "HuggingFaceTB/SmolLM2-135M"
# model_name = "Luigi/SmolLM2-360M-Instruct-TaiwanChat"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_advice(prompt, tokenizer=tokenizer, model=model, max_new_tokens=160):
    """
    使用小型 LLM 產生繁體中文建議。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    advice = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return advice.strip()

# ----------  LLM 建議 ----------
prompt = (
    f"你是一位節能顧問，請依據以下數據條列3點繁體中文建議。\n"
    # f"- 本月用水 {curr_water:.2f} ML，預估下月 {next_water:.2f} ML，{trend_water} {water_pct:+.1f}%\n"
    f"- 本月用電 {curr_elec:.2f} kWh，預估下月 {next_elec:.2f} kWh，{trend_elec} {elec_pct:+.1f}%\n"
    f"可以從一些日常習慣與常見的電器使用方式來建議。"
)

advice = generate_advice(prompt)
print("\n============ 建議 ============")
print(advice)
print("================================")

# ========== 預測部分已註解 ==========
# # ----------  預測 ----------
# next_water = predict_next_month(series_water, months_water, water_model_path)
# next_elec  = predict_next_month(series_elec, months_elec, elec_model_path)
#
# # ----------  輸出 ----------
# print(f"🔹 本月用水量：{series_water[-1]:,.2f} ML")
# print(f"🔹 本月用電量：{series_elec[-1]:,.2f} kWh")
# print(f"🔹 下個月預估用水量：{next_water:,.2f} ML")
# print(f"🔹 下個月預估用電量：{next_elec:,.2f} kWh")