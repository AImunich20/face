from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage, MessagingApiBlob
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent,
    ImageMessageContent, VideoMessageContent
)
from linebot.v3.exceptions import InvalidSignatureError

import os, re, csv, shutil, requests
from datetime import datetime

from dataset import build_dataset_from_csv
from picture_model import process_image
from train import train_model_from_csv

# ===== CONFIG =====
CHANNEL_ACCESS_TOKEN = "9pOzQ0MC0T1rl9ybqhyuy/Gck33gWh7fOAaKQOMDxKUa0sAeBNSqtB2EdCWjqIMc8NjMy3oF0CVJq6MR5J1gVL9RK3qhvVMslQG3UB+o1tbnZjKPNghtZKc3bE5uKYzt0EmOZ5/MK7rEfTiHgX7h/wdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET = "d212cc6ec4d574800b33b92a7962f173"

app = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

CSV_FILE = "users.csv"
USER_FOLDER = "user"
os.makedirs(USER_FOLDER, exist_ok=True)

# ===== STATE =====
user_states = {}

# ================== ROUTE ==================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# ================== UTIL ==================
def reply(event, text):
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=text)]
            )
        )

def send_line_message(user_id, text):
    requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        },
        json={"to": user_id, "messages": [{"type": "text", "text": text}]}
    )

def extract_code(text):
    match = re.search(r"\((.*?)\)", text)
    return match.group(1) if match else None

def get_users():
    if not os.path.exists(CSV_FILE):
        return []

    with open(CSV_FILE, encoding="utf-8") as f:
        return [row[0] for i, row in enumerate(csv.reader(f)) if i > 0]

# ================== MODE ==================
def detect_mode(text):
    if "เพิ่ม user" in text:
        return "ADD"
    if "ลบ user" in text:
        return "DELETE"
    return "NORMAL"

# ================== ADD USER ==================
def add_user_mode(event, user_id, text):
    state = user_states.get(user_id, {})

    if state.get("mode") == "ASK_NEXT":
        if text == "ต่อ":
            user_states[user_id] = {"mode": "ADD"}
            return reply(event, "ขอชื่อ")

        if text == "เทรน":
            send_line_message(user_id, "กำลังทำ dataset ...")
            build_dataset_from_csv()
            send_line_message(user_id, "dataset เสร็จแล้ว")
            send_line_message(user_id, "กำลังทำ Training ...")
            train_model_from_csv()
            send_line_message(user_id, "Training เสร็จแล้ว")
            user_states.pop(user_id)
            return

        return reply(event, "พิมพ์ ต่อ / เทรน")

    if state.get("mode") == "ADD":
        user_states[user_id] = {
            "mode": "WAIT_VIDEO",
            "name": text
        }
        return reply(event, "ส่งวิดีโอมา")

    user_states[user_id] = {"mode": "ADD"}
    reply(event, "ขอชื่อ")

# ================== DELETE ==================
def delete_user_mode(event, user_id, text):
    state = user_states.get(user_id)

    if state and state.get("mode") == "DELETE":
        users = get_users()

        if text not in users:
            return reply(event, "ไม่พบ")

        # ลบจาก CSV
        rows = []
        with open(CSV_FILE, encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                if row[0] != text:
                    rows.append(row)

        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

        user_states.pop(user_id)
        return reply(event, f"ลบ {text} แล้ว")

    users = get_users()
    user_states[user_id] = {"mode": "DELETE"}

    return reply(event, "จะลบใคร?\n" + "\n".join(users))

# ================== IMAGE ==================
def handle_image(event):
    user_id = event.source.user_id
    state = user_states.get(user_id)

    if not state or state.get("mode") != "WAIT_IMAGE":
        return

    message_id = event.message.id

    res = requests.get(
        f"https://api-data.line.me/v2/bot/message/{message_id}/content",
        headers={"Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"}
    )

    if res.status_code != 200:
        return send_line_message(user_id, "โหลดรูปไม่ได้")

    os.makedirs("day_face_pic", exist_ok=True)
    path = f"day_face_pic/{user_id}.jpg"

    with open(path, "wb") as f:
        f.write(res.content)

    results, output_path = process_image(path)

    if results:
        msg = "\n".join([f"{r['name']} ({r['face_confidence']})" for r in results])
    else:
        msg = "ไม่พบหน้า"

    send_line_message(user_id, msg)
    user_states.pop(user_id)

# ================== VIDEO ==================
@handler.add(MessageEvent, message=VideoMessageContent)
def handle_video(event):
    user_id = event.source.user_id
    state = user_states.get(user_id)

    if not state or state.get("mode") != "WAIT_VIDEO":
        return

    name = state["name"]

    with ApiClient(configuration) as api_client:
        blob = MessagingApiBlob(api_client)
        content = blob.get_message_content(event.message.id)

    path = f"user/{name}.mp4"
    with open(path, "wb") as f:
        f.write(content)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if os.stat(CSV_FILE).st_size == 0:
            w.writerow(["name", "time", "path"])
        w.writerow([name, datetime.now(), path])

    user_states[user_id] = {"mode": "ASK_NEXT"}
    reply(event, "เพิ่มอีกไหม (ต่อ / เทรน)")

# ================== EVENTS ==================
@handler.add(MessageEvent, message=ImageMessageContent)
def img(event):
    handle_image(event)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text

    state = user_states.get(user_id)
    mode = detect_mode(text)

    if state:
        m = state.get("mode")

        if m in ["ADD", "ASK_NEXT"]:
            return add_user_mode(event, user_id, text)

        if m == "DELETE":
            return delete_user_mode(event, user_id, text)

    if mode == "ADD":
        return add_user_mode(event, user_id, text)

    if mode == "DELETE":
        return delete_user_mode(event, user_id, text)

    if "เช็คชื่อ" in text:
        user_states[user_id] = {"mode": "WAIT_IMAGE"}
        return reply(event, "ส่งรูปมา")

# ================== RUN ==================
if __name__ == "__main__":
    app.run(port=5000, debug=True)