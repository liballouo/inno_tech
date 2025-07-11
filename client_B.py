import socket
import json
import numpy as np
from custum_predict import predict_custom_next

# 指定client name: B: LSTM預測板; C: 筆電
CLIENT_NAME = "B"
MODEL_PATH = "predict_lstm.pt"  # 根據實際模型路徑調整

# 這裡直接放入你要預測的資料
series = np.array([
    7458.12, 6569.6, 7717.07, 7384.67, 8225.05, 7948.97, 
    8747.56, 8891.44, 9639.7, 9433.45, 8708.31, 7646.66, 
    7472.58, 6828.5, 7527.25, 6746.62, 7631.52, 8233.8, 
    9675.76, 9822.37, 10645.8, 9380.99, 8587.96, 7926.62, 
    7605.3, 6882.43, 6979.24, 6957.39, 7945.1, 8157.05, 
    9696.06, 9661.71, 10285.28, 9204.46, 8788.11, 7757.88
])
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]*3)

# 啟動時先做一次預測
predict_val = float(predict_custom_next(series, months, MODEL_PATH))
current_month = int(series[-1])
print("已完成預測，預測值：", predict_val)

HOST = '192.168.1.6'  # 換成A的IP
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(CLIENT_NAME.encode('utf-8'))

    while True:
        try:
            data = s.recv(4096)
            if not data:
                print("連線中斷")
                break
            req = json.loads(data.decode('utf-8'))
            print("收到A資料：", req)
            # 準備回傳資料
            resp = {
                "current_month": current_month,
                "predict": predict_val,
                "P1": req.get("P1", None)
            }
            s.sendall(json.dumps(resp).encode('utf-8'))
            print("已回傳：", resp)
        except Exception as e:
            print("接收或處理資料時發生錯誤：", e)
            break