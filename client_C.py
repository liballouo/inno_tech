import socket

CLIENT_NAME = "C"
HOST = '192.168.1.6'  # 換成A的IP
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(CLIENT_NAME.encode('utf-8'))
    print("已連線，等待資料...")
    while True:
        try:
            data = s.recv(4096)
            if not data:
                print("連線中斷")
                break
            print("收到資料：", data.decode())
        except Exception as e:
            print("接收錯誤：", e)
            break