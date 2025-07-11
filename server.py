import socket
import threading
import time
import json
from queue import Queue

# LLM相關import
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_advice(prompt, tokenizer, model, max_new_tokens=160):
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

# 載入LLM模型（只載入一次）
model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

clients = {}
lock = threading.Lock()
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

            with lock:
                if client_name == "B":
                    print(f"B 回傳資料: {msg}")
                    # 將B的回傳資料放入佇列，給C
                    b_to_c_queue.put(msg)
                elif client_name == "A":
                    for target in ["B", "C"]:
                        if target in clients:
                            clients[target].sendall(f"來自A: {msg}".encode())
                elif client_name == "C":
                    # 處理C傳來的即時資料
                    print(f"C 傳來即時資料: {msg}")
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
        client_name = conn.recv(1024).decode()
        with lock:
            clients[client_name] = conn
        threading.Thread(target=handle_client, args=(conn, addr, client_name), daemon=True).start()

def periodic_send_to_B_and_forward_to_C():
    i = 0
    while True:
        time.sleep(5)
        with lock:
            if "B" in clients and "C" in clients:
                data = {"P1": 100 + i, "P2": 200 + i}
                try:
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
                        # 解析B的回傳資料，組LLM prompt
                        try:
                            b_data = json.loads(msg)
                            cm1 = b_data.get("current_month_1")
                            cm2 = b_data.get("current_month_2")
                            p1 = b_data.get("predict_1")
                            p2 = b_data.get("predict_2")
                            prompt = (
                                f"你是一位節能顧問，請依據以下數據條列3點繁體中文建議。\n"
                                f"- 本月用電1 {cm1} kWh，預估下月1 {p1} kWh\n"
                                f"- 本月用電2 {cm2} kWh，預估下月2 {p2} kWh\n"
                                f"可以從一些日常習慣與常見的電器使用方式來建議。"
                            )
                            advice = generate_advice(prompt, tokenizer, model)
                            b_data["llm_advice"] = advice
                            send_msg = json.dumps(b_data, ensure_ascii=False)
                        except Exception as e:
                            send_msg = json.dumps({"error": f"LLM處理失敗: {e}", "raw": msg}, ensure_ascii=False)
                        clients["C"].sendall(send_msg.encode('utf-8'))
                        print(f"已將B的資料與LLM建議轉發給C: {send_msg}")
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

    threading.Thread(target=periodic_send_to_B_and_forward_to_C, daemon=True).start()
    accept_clients(server_socket)
