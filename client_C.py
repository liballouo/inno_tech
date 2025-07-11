import socket
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

CLIENT_NAME = "C"
HOST = '192.168.1.6'  # 換成A的IP
PORT = 5000

# ====== LLM 載入 ======
model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_advice(prompt, tokenizer=tokenizer, model=model, max_new_tokens=160):
    """
    使用小型 LLM 產生繁體中文建議。
    """
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

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(CLIENT_NAME.encode('utf-8'))
    print("C 已連線，等待資料...")
    while True:
        try:
            data = s.recv(4096)
            if not data:
                print("連線中斷")
                break
            msg = data.decode()
            print("收到資料：", msg)
            try:
                req = json.loads(msg)
                # 解析資料
                current_month = req.get("current_month")
                predict = req.get("predict")
                # 計算趨勢與百分比
                if current_month is not None and predict is not None:
                    trend = "上升" if predict > current_month else "下降"
                    pct = ((predict - current_month) / current_month) * 100 if current_month != 0 else 0
                    # 組 prompt
                    prompt = (
                        f"你是一位節能顧問，請依據以下數據條列3點繁體中文建議。\n"
                        f"- 本月用電 {current_month:.2f} kWh，預估下月 {predict:.2f} kWh，{trend} {pct:+.1f}%\n"
                        f"可以從一些日常習慣與常見的電器使用方式來建議。"
                    )
                    advice = generate_advice(prompt)
                    print("\n============ LLM 建議 ============")
                    print(advice)
                    print("==================================\n")
                else:
                    print("收到的資料缺少 current_month 或 predict 欄位")
            except Exception as e:
                print("解析或產生建議時發生錯誤：", e)
        except Exception as e:
            print("接收錯誤：", e)
            break