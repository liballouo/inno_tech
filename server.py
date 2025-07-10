import socket
import threading

# 儲存 client 連線
clients = {}
lock = threading.Lock()

def handle_client(conn, addr, client_name):
    print(f"{client_name} 已連線：{addr}")
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            msg = data.decode()
            print(f"收到 {client_name} 訊息：{msg}")

            # 根據來源分發訊息
            with lock:
                if client_name == "B":
                    # B 傳來的訊息發給 C
                    if "C" in clients:
                        clients["C"].sendall(f"來自B: {msg}".encode())
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

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 5000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"伺服器啟動於 {HOST}:{PORT}，等待連線...")

    accept_clients(server_socket)
