import socket
import threading
import time
import json
from queue import Queue
import os

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
# 新增一個佇列用於C->B的即時資料
c_to_b_queue = Queue()
# B->C的回傳資料佇列（如需後續LLM處理可保留）
b_to_c_queue = Queue()

last_b_returned = threading.Event()  # 用於同步B回傳

LLM_STATUS_PATH = "LLM_status.json"
LLM_RESULT_PATH = "LLM_result.json"
DATA_PATH = "data.json"

# 啟動時自動建立檔案與初始值
init_files = [
    (DATA_PATH, {"daily_p1": 0, "daily_p2": 0, "monthly_p1": 0, "monthly_p2": 0, "predict_p1": 0, "predict_p2": 0}),
    (LLM_STATUS_PATH, {"status": "生成完畢"}),
    (LLM_RESULT_PATH, {"LLM_advice": "尚未產生建議"})
]
for path, default in init_files:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False)
        print(f"已建立初始檔案: {path}")

# 新增：定時檢查LLM_status.json並執行LLM建議

def llm_status_watcher():
    while True:
        time.sleep(1)
        try:
            if not os.path.exists(LLM_STATUS_PATH):
                continue
            with open(LLM_STATUS_PATH, "r", encoding="utf-8") as f:
                status_data = json.load(f)
            status = status_data.get("status", "")
            if status == "請生成":
                # 先設為生成中
                status_data["status"] = "生成中"
                with open(LLM_STATUS_PATH, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, ensure_ascii=False)
                # 讀取data.json作為LLM資料來源
                if not os.path.exists(DATA_PATH):
                    print("找不到data.json，無法執行LLM建議生成")
                    continue
                with open(DATA_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    daily_p1 = data.get("daily_p1", 0)
                    daily_p2 = data.get("daily_p2", 0)
                    monthly_p1 = data.get("monthly_p1", 0)
                    monthly_p2 = data.get("monthly_p2", 0)
                    predict_p1 = data.get("predict_p1", 0)
                    predict_p2 = data.get("predict_p2", 0)
                    prompt = (
                        f"你是一位節能顧問，請依據以下數據條列3點繁體中文建議。\n"
                        f"本日累積用電 {daily_p1+daily_p2} kWh；本月累積用電 {monthly_p1+monthly_p2} kWh； 本日預測用電 {(predict_p1+predict_p2)/30} kWh； 本月預測用電 {predict_p1+predict_p2} kWh \n"
                        f"可以從一些日常習慣與常見的電器使用方式來建議。"
                    )
                    advice = generate_advice(prompt, tokenizer, model)
                    # 寫入LLM_result.json
                    with open(LLM_RESULT_PATH, "w", encoding="utf-8") as f:
                        json.dump({"LLM_advice": advice}, f, ensure_ascii=False)
                    # 設為生成完畢
                    status_data["status"] = "生成完畢"
                    with open(LLM_STATUS_PATH, "w", encoding="utf-8") as f:
                        json.dump(status_data, f, ensure_ascii=False)
                    print("LLM建議已生成並寫入LLM_result.json")
        except Exception as e:
            print(f"LLM狀態檢查/生成時發生錯誤: {e}")


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
                    last_b_returned.set()  # 標記B已回傳
                    try:
                        b_data = json.loads(msg)
                        predict_update = {k: b_data[k] for k in ["predict_p1", "predict_p2"] if k in b_data}
                        # 讀取現有data.json，更新predict欄位
                        if os.path.exists(DATA_PATH):
                            with open(DATA_PATH, "r", encoding="utf-8") as f:
                                data_json = json.load(f)
                        else:
                            data_json = {}
                        data_json.update(predict_update)
                        with open(DATA_PATH, "w", encoding="utf-8") as f:
                            json.dump(data_json, f, ensure_ascii=False)
                        print(f"已將B的預測資料寫入data.json: {predict_update}")
                    except Exception as e:
                        print(f"解析B資料失敗: {e}")
                elif client_name == "C":
                    try:
                        c_data = json.loads(msg)
                        filtered = {k: c_data[k] for k in ["daily_p1", "daily_p2", "monthly_p1", "monthly_p2"] if k in c_data}
                        # 讀取現有data.json，更新daily/monthly欄位
                        if os.path.exists(DATA_PATH):
                            with open(DATA_PATH, "r", encoding="utf-8") as f:
                                data_json = json.load(f)
                        else:
                            data_json = {}
                        data_json.update(filtered)
                        with open(DATA_PATH, "w", encoding="utf-8") as f:
                            json.dump(data_json, f, ensure_ascii=False)
                        print(f"已更新data.json: {filtered}")
                    except Exception as e:
                        print(f"解析C資料失敗: {e}")
                
                elif client_name == "A":
                    for target in ["B", "C"]:
                        if target in clients:
                            clients[target].sendall(f"來自A: {msg}".encode())
        except Exception as e:
            print(f"{client_name} 連線異常：{e}")
            break
    with lock:
        del clients[client_name]
    conn.close()
    print(f"{client_name} 離線")

def datajson_to_B_sender():
    while True:
        time.sleep(0.5)
        with lock:
            if "B" in clients:
                # 讀取data.json
                if os.path.exists(DATA_PATH):
                    with open(DATA_PATH, "r", encoding="utf-8") as f:
                        data_json = json.load(f)
                    daily_p1 = data_json.get("daily_p1", 0)
                    daily_p2 = data_json.get("daily_p2", 0)
                    monthly_p1 = data_json.get("monthly_p1", 0)
                    monthly_p2 = data_json.get("monthly_p2", 0)
                    # 檢查四個值都不是0
                    if all([daily_p1, daily_p2, monthly_p1, monthly_p2]):
                        # 只有收到B的回傳值後才傳送
                        if last_b_returned.is_set() or (data_json.get("predict_p1") == 0 and data_json.get("predict_p2") == 0):
                            send_data = {
                                "daily_p1": daily_p1,
                                "daily_p2": daily_p2,
                                "monthly_p1": monthly_p1,
                                "monthly_p2": monthly_p2
                            }
                            try:
                                clients["B"].sendall(json.dumps(send_data).encode('utf-8'))
                                print(f"已將data.json資料傳給B: {send_data}")
                                last_b_returned.clear()  # 等待下次B回傳
                            except Exception as e:
                                print(f"傳送資料給B失敗: {e}")

def accept_clients(server_socket):
    while True:
        conn, addr = server_socket.accept()
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

    threading.Thread(target=llm_status_watcher, daemon=True).start()
    threading.Thread(target=datajson_to_B_sender, daemon=True).start()
    accept_clients(server_socket)
