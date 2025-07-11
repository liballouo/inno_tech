import socket
import json
import time

CLIENT_NAME = "C"
HOST = '192.168.137.1'  # 換成筆電的IP
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(CLIENT_NAME.encode('utf-8'))
    print("C 已連線，開始傳送即時資料...")
    i = 0
    while True:
        # 模擬即時資料
        data = {
            "P1": 100 + i,
            "P2": 200 + i
        }
        s.sendall(json.dumps(data).encode('utf-8'))
        print("已傳送資料：", data)
        i += 1
        time.sleep(5)  # 每5秒傳送一次