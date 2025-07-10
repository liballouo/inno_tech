# # socket_client.py
# import socket

# HOST = '192.168.1.29'  # 換成你板子的 IP（可用 `ip a` 查）
# PORT = 8787

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect((HOST, PORT))
#     s.sendall(b'Hello, iMX93!')
#     data = s.recv(1024)

# print(f"Received from server: {data.decode()}")

import socket
import json

# 準備要傳送的資料
data = {
    "series": [7458.12, 6569.6, 7717.07, 7384.67, 8225.05, 7948.97, 
                8747.56, 8891.44, 9639.7, 9433.45, 8708.31, 7646.66, 
                7472.58, 6828.5, 7527.25, 6746.62, 7631.52, 8233.8, 
                9675.76, 9822.37, 10645.8, 9380.99, 8587.96, 7926.62, 
                7605.3, 6882.43, 6979.24, 6957.39, 7945.1, 8157.05, 
                9696.06, 9661.71, 10285.28, 9204.46, 8788.11, 7757.88],
    "months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]*3
}

HOST = '192.168.1.29'  # 換成你板子的 IP（可用 `ip a` 查）
PORT = 8787

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(json.dumps(data).encode('utf-8'))
    s.shutdown(socket.SHUT_WR)  # 告訴 server 資料已傳送完畢
    response = b""
    while True:
        packet = s.recv(4096)
        if not packet:
            break
        response += packet
    result = json.loads(response.decode('utf-8'))
    print("Server 回傳：", result)