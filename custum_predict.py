# -*- coding: utf-8 -*-
"""
根據自定義輸入，使用已訓練好的 LSTM 模型進行單步預測，並輸出結果。
現在改為透過 socket 接收輸入資料。

輸入範例: {"series": [100, 110, 120, ...], "months": [1, 2, 3, ...]}
輸出範例: {"next_val": 123.45}
"""

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import socket
import json

# 載入模型架構
from design_forecast_lstm import EnergyPredictorLSTM, multi_step_forecast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(model_path):
    model = EnergyPredictorLSTM(input_size=3, hidden_size=256, num_layers=4, dropout=0.3).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_custom_next(series, months, model_path):
    """
    series: numpy array, shape (N,)
    months: numpy array, shape (N,)
    model_path: str, 已訓練模型路徑
    回傳: 預測下一個月的值 (float)
    """
    scaler = MinMaxScaler()
    scaler.fit(series.reshape(-1, 1))
    model = load_trained_model(model_path)
    next_val = multi_step_forecast(model, scaler, history_data=series, history_months=months, steps=1)[0]
    return next_val

def start_socket_server(host='192.168.1.29', port=8787, model_path="predict_lstm.pt"):
    """
    啟動 socket server，等待接收 JSON 格式的 series 和 months，並回傳預測結果。
    輸入格式: {"series": [...], "months": [...]}
    回傳格式: {"next_val": ...}
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        print(f"Socket server listening on {host}:{port} ...")
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                data = b""
                while True:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data += packet
                try:
                    req = json.loads(data.decode("utf-8"))
                    series = np.array(req["series"])
                    months = np.array(req["months"])
                    next_val = predict_custom_next(series, months, model_path)
                    resp = {"next_val": float(next_val)}
                except Exception as e:
                    resp = {"error": str(e)}
                conn.sendall(json.dumps(resp).encode("utf-8"))

if __name__ == "__main__":
    # ====== 使用者自定義輸入區（已註解） ======
    # series = np.array([116.22, 118.45, 98.03, 130.7, 73.15, 33.05, 57.18, 87.26, 93.57, 175.63, 158.92,
    #                    139.24, 96.28, 90.82, 85.07, 88.23, 65.82, 70.41, 90.82, 88.23, 92.83, 136.52, 122.43, 
    #                    118.99, 122.16,  83.38,  82.82, 105.26,  96.39,  86.15, 94.73,  88.08,  96.95, 134.07, 124.65, 111.63])
    # months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]*3)
    # model_path = "predict_lstm.pt"
    # next_val = predict_custom_next(series, months, model_path)
    # print(f"自定義輸入下，下一個月預測值：{next_val:.2f}")

    # ====== 改為 socket 伺服器接收輸入 ======
    start_socket_server()