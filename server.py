import socket
import threading
import time
import json

# 儲存 client 連線
clients = {}
lock = threading.Lock()

# 新增一個佇列用於B的回傳資料
from queue import Queue
b_to_c_queue = Queue()

def handle_client(conn, addr, client_name):
    print(f"{client_name} 已連線：{addr}")
    while True:
        try:
            data = conn.recv(4096)
            if not data:
                break
            msg = data.decode()
            print(f"收到 {client_name} 訊息：{msg}")

            # 根據來源分發訊息
            with lock:
                if client_name == "B":
                    print(f"B 回傳資料: {msg}")
                    # 將B的回傳資料放入佇列，給C
                    b_to_c_queue.put(msg)
                elif client_name == "A":
                    # A 自己發的訊息給 B 和 C
                    for target in ["B", "C"]:
                        if target in clients:
                            clients[target].sendall(f"來自A: {msg}".encode())
                elif client_name == "C":
                    # C 傳來的訊息可忽略或自訂處理
                    pass
        except Exception as e:
            print(f"{client_name} 連線異常：{e}")
            break
    with lock:
        del clients[client_name]
    conn.close()
    print(f"{client_name} 離線")

def accept_clients(server_socket):
    while True:
        conn, addr = server_socket.accept()
        # 連線後，先接收 client 名稱
        client_name = conn.recv(1024).decode()
        with lock:
            clients[client_name] = conn
        threading.Thread(target=handle_client, args=(conn, addr, client_name), daemon=True).start()

def periodic_send_to_B_and_forward_to_C():
    i = 0
    while True:
        time.sleep(5)
        with lock:
            # 必須 B、C 都連線才傳送
            if "B" in clients and "C" in clients:
                data = {"P1": 100 + i}
                try:
                    # 傳送資料給B
                    clients["B"].sendall(json.dumps(data).encode('utf-8'))
                    print(f"已發送資料給B: {data}")
                except Exception as e:
                    print(f"發送給B失敗: {e}")
            else:
                print("等待B、C都連線中...")
        # 檢查是否有B的回傳資料要給C
        try:
            while not b_to_c_queue.empty():
                msg = b_to_c_queue.get()
                with lock:
                    if "C" in clients:
                        clients["C"].sendall(msg.encode('utf-8'))
                        print(f"已將B的資料轉發給C: {msg}")
        except Exception as e:
            print(f"轉發給C失敗: {e}")
        i += 1

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 5000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"伺服器啟動於 {HOST}:{PORT}，等待連線...")

    # 啟動定時傳送資料給B的thread
    threading.Thread(target=periodic_send_to_B_and_forward_to_C, daemon=True).start()

    accept_clients(server_socket)
