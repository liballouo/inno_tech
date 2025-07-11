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

LLM_STATUS_PATH = "LLM_status.json"
LLM_RESULT_PATH = "LLM_result.json"
DATA_PATH = "data.json"

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
    buffer = ""
    while True:
        try:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data.decode()
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if not line.strip():
                    continue
                try:
                    msg_json = json.loads(line)
                    # 處理msg_json
                    print(f"收到 {client_name} 訊息：{msg_json}")

                    with lock:
                        if client_name == "C":
                            # C 傳來即時資料，放入佇列給B
                            try:
                                c_data = msg_json
                                # 僅保留需要的欄位
                                filtered = {k: c_data[k] for k in ["daily_p1", "daily_p2", "monthly_p1", "monthly_p2"] if k in c_data}
                                c_to_b_queue.put(json.dumps(filtered, ensure_ascii=False))
                                print(f"已將C的即時資料放入佇列給B: {filtered}")
                            except Exception as e:
                                print(f"解析C資料失敗: {e}")
                        elif client_name == "B":
                            # B 回傳資料，放入佇列給C或後續LLM
                            print(f"B 回傳資料: {msg_json}")
                            b_to_c_queue.put(json.dumps(msg_json, ensure_ascii=False)) # 確保是單一JSON
                            # 新增：將B回傳資料寫入data.json
                            try:
                                with open("data.json", "w", encoding="utf-8") as f:
                                    f.write(line) # 寫入原始行
                                print("已將B回傳資料寫入data.json")
                            except Exception as e:
                                print(f"寫入data.json失敗: {e}")
                        elif client_name == "A":
                            for target in ["B", "C"]:
                                if target in clients:
                                    clients[target].sendall(f"來自A: {msg_json}".encode())
                        # elif client_name == "C":
                        #     # 處理C傳來的即時資料
                        #     print(f"C 傳來即時資料: {msg_json}")
                except Exception as e:
                    print("解析JSON失敗：", e)
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

def c_to_b_forwarder():
    # 將C的即時資料轉發給B
    while True:
        time.sleep(0.5)
        with lock:
            if "B" in clients:
                try:
                    while not c_to_b_queue.empty():
                        msg = c_to_b_queue.get()
                        clients["B"].sendall(msg.encode('utf-8'))
                        print(f"已將C的即時資料轉發給B: {msg}")
                except Exception as e:
                    print(f"發送給B失敗: {e}")
            else:
                print("等待B、C都連線中...")
        # # 檢查是否有B的回傳資料要給C
        # try:
        #     while not b_to_c_queue.empty():
        #         msg = b_to_c_queue.get()
        #         with lock:
        #             if "C" in clients:
        #                 # 解析B的回傳資料，組LLM prompt
        #                 try:
        #                     b_data = json.loads(msg)
        #                     # cm1 = b_data.get("current_month_1")
        #                     # cm2 = b_data.get("current_month_2")
        #                     daily_p1 = b_data.get("daily_p1")
        #                     daily_p2 = b_data.get("daily_p2")
        #                     monthly_p1 = b_data.get("monthly_p1")
        #                     monthly_p2 = b_data.get("monthly_p2")
        #                     predict_p1 = b_data.get("predict_1")
        #                     predict_p2 = b_data.get("predict_2")
        #                     prompt = (
        #                         f"你是一位節能顧問，請依據以下數據條列3點繁體中文建議。\n"
        #                         # f"- 上個月用電1 {cm1} kWh，預估本月1 {p1} kWh\n"
        #                         # f"- 上個月用電2 {cm2} kWh，預估本月2 {p2} kWh\n"
        #                         f"本日累積用電 {daily_p1+daily_p2} kWh；本月累積用電 {monthly_p1+monthly_p2} kWh； 本日預測用電 {(predict_p1+predict_p2)/30} kWh； 本月預測用電 {predict_p1+predict_p2} kWh \n"
        #                         f"可以從一些日常習慣與常見的電器使用方式來建議。"
        #                     )
        #                     advice = generate_advice(prompt, tokenizer, model)
        #                     b_data["llm_advice"] = advice
        #                     send_msg = json.dumps(b_data, ensure_ascii=False)
        #                 except Exception as e:
        #                     send_msg = json.dumps({"error": f"LLM處理失敗: {e}", "raw": msg}, ensure_ascii=False)
        #                 clients["C"].sendall(send_msg.encode('utf-8'))
        #                 print(f"已將B的資料與LLM建議轉發給C: {send_msg}")
        # except Exception as e:
        #     print(f"轉發給C失敗: {e}")
        # i += 1

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 5000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"伺服器啟動於 {HOST}:{PORT}，等待連線...")

    threading.Thread(target=llm_status_watcher, daemon=True).start()
    threading.Thread(target=c_to_b_forwarder, daemon=True).start()
    accept_clients(server_socket)
