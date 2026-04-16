from flask import Flask, request, abort, send_from_directory
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
import json

from dataset import build_dataset_from_csv
from picture_model import process
from train import train_face_recognition
from sheet import send_to_google_sheet

# ===== CONFIG =====
CHANNEL_ACCESS_TOKEN = "9pOzQ0MC0T1rl9ybqhyuy/Gck33gWh7fOAaKQOMDxKUa0sAeBNSqtB2EdCWjqIMc8NjMy3oF0CVJq6MR5J1gVL9RK3qhvVMslQG3UB+o1tbnZjKPNghtZKc3bE5uKYzt0EmOZ5/MK7rEfTiHgX7h/wdB04t89/1O/w1cDnyilFU="
CHANNEL_SECRET = "d212cc6ec4d574800b33b92a7962f173"

app = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

CSV_FILE = "users.csv"
USER_FOLDER = "user"
os.makedirs(USER_FOLDER, exist_ok=True)

RESULTS_DIR = os.path.join(os.getcwd(), 'day_face_result')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== STATE =====
user_states = {}

@app.route('/results/<path:filename>')
def serve_results(filename):
    clean_filename = os.path.basename(filename)
    return send_from_directory('day_face_result', clean_filename)

# ================== ROUTE ==================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    print(f"[WEBHOOK] Received webhook callback")

    try:
        handler.handle(body, signature)
        print(f"[WEBHOOK] Webhook processed successfully")
    except InvalidSignatureError:
        print(f"[WEBHOOK] Invalid signature")
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

def send_line_image(user_id, image_url):
    requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "to": user_id, 
            "messages": [
                {
                    "type": "image",
                    "originalContentUrl": image_url,
                    "previewImageUrl": image_url
                }
            ]
        }
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
    print(f"[ADD_USER] user_id={user_id}, text={text}")
    state = user_states.get(user_id, {})

    if state.get("mode") == "ASK_NEXT":
        if text == "ต่อ":
            print(f"[ADD_USER] Continuing add mode")
            user_states[user_id] = {"mode": "ADD"}
            return reply(event, "ขอชื่อ")

        if text == "เทรน":
            print(f"[ADD_USER] Starting training for user_id={user_id}")
            send_line_message(user_id, "กำลังทำ dataset ...")
            build_dataset_from_csv()
            print(f"[ADD_USER] Dataset built")
            send_line_message(user_id, "dataset เสร็จแล้ว")
            send_line_message(user_id, "กำลังทำ Training ...")
            train_face_recognition()
            print(f"[ADD_USER] Training completed")
            send_line_message(user_id, "Training เสร็จแล้ว")
            user_states.pop(user_id)
            return

        return reply(event, "พิมพ์ ต่อ / เทรน")

    if state.get("mode") == "ADD":
        print(f"[ADD_USER] Waiting for video, name={text}")
        user_states[user_id] = {
            "mode": "WAIT_VIDEO",
            "name": text
        }
        return reply(event, "ส่งวิดีโอมา")

    print(f"[ADD_USER] Starting add user mode")
    user_states[user_id] = {"mode": "ADD"}
    reply(event, "ขอชื่อ")

# ================== DELETE ==================
def delete_user_mode(event, user_id, text):
    print(f"[DELETE_USER] user_id={user_id}, text={text}")
    state = user_states.get(user_id)

    if not state or state.get("mode") != "DELETE":
        users = get_users()
        print(f"[DELETE_USER] Available users: {users}")
        user_states[user_id] = {"mode": "DELETE"}
        return reply(event, "จะลบใคร?\n" + "\n".join(users))

    users = get_users()

    if text not in users:
        print(f"[DELETE_USER] User {text} not found")
        return reply(event, "❌ ไม่พบ user")

    video_path = f"user/{text}.mp4"
    dataset_path = f"dataset/{text}"

    trash_video_dir = "user/trash"
    trash_dataset_dir = "dataset_trash"

    os.makedirs(trash_video_dir, exist_ok=True)
    os.makedirs(trash_dataset_dir, exist_ok=True)

    if os.path.exists(video_path):
        trash_video_path = os.path.join(trash_video_dir, f"{text}.mp4")
        if os.path.exists(trash_video_path):
            os.remove(trash_video_path)
        shutil.move(video_path, trash_video_path)

    if os.path.exists(dataset_path):
        trash_dataset_path = os.path.join(trash_dataset_dir, text)
        if os.path.exists(trash_dataset_path):
            shutil.rmtree(trash_dataset_path)
        shutil.move(dataset_path, trash_dataset_path)

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

    print(f"[DELETE_USER] Successfully deleted user: {text}")
    user_states.pop(user_id, None)

    return reply(event, f"🗑️ ลบ {text} แล้ว (video + dataset + csv)")

# ================== IMAGE ==================
def handle_image(event):
    user_id = event.source.user_id
    print(f"[IMAGE] Received image from user_id={user_id}")
    state = user_states.get(user_id)

    if not state or state.get("mode") != "WAIT_IMAGE":
        print(f"[IMAGE] User not in WAIT_IMAGE mode, current state: {state}")
        return

    message_id = event.message.id
    print(f"[IMAGE] message_id={message_id}")

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

    print(f"[IMAGE] Processing image: {path}")

    results,result_img_path,result_json_path = process(path)
    print(f"[IMAGE] Detection results: {results}")
    print(f"[IMAGE] Result image path: {result_img_path}")
    print(f"[IMAGE] Result JSON path: {result_json_path}")
    if results:
        msg = "พบบุคคล:\n- " + "\n- ".join(results)
        print(f"[IMAGE] Found {len(results)} persons: {results}")
    else:
        msg = "ไม่พบบุคคลในภาพ"
        print(f"[IMAGE] No persons detected")

    send_line_message(user_id, msg)
    pitunnelURL = "https://linethreera-natthanat.as2.pitunnel.net"
    file_name_only = os.path.basename(result_img_path)
    full_url = f"{pitunnelURL}/results/{file_name_only}"
    print(f"[IMAGE] Sending result image URL: {full_url}")

    send_line_image(user_id, full_url)
    with open(result_json_path, 'r', encoding='utf-8') as f:
        local_data = json.load(f)
        sheetst = send_to_google_sheet(local_data)
    send_line_message(user_id, sheetst)

# ================== VIDEO ==================
@handler.add(MessageEvent, message=VideoMessageContent)
def handle_video(event):
    user_id = event.source.user_id
    print(f"[VIDEO] Received video from user_id={user_id}")
    state = user_states.get(user_id)

    if not state or state.get("mode") != "WAIT_VIDEO":
        print(f"[VIDEO] User not in WAIT_VIDEO mode")
        return

    name = state["name"]
    print(f"[VIDEO] Saving video for user name: {name}")

    with ApiClient(configuration) as api_client:
        blob = MessagingApiBlob(api_client)
        content = blob.get_message_content(event.message.id)

    path = f"user/{name}.mp4"
    print(f"[VIDEO] Writing video to: {path}")
    with open(path, "wb") as f:
        f.write(content)
    print(f"[VIDEO] Video saved, size: {len(content)} bytes")

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if os.stat(CSV_FILE).st_size == 0:
            w.writerow(["name", "time", "path"])
        w.writerow([name, datetime.now(), path])
    print(f"[VIDEO] CSV updated for user: {name}")

    user_states[user_id] = {"mode": "ASK_NEXT"}
    print(f"[VIDEO] Asking user to continue or train")
    reply(event, "เพิ่มอีกไหม (ต่อ / เทรน)")

# ================== EVENTS ==================
@handler.add(MessageEvent, message=ImageMessageContent)
def img(event):
    handle_image(event)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text
    print(f"[TEXT_MESSAGE] user_id={user_id}, text={text}")

    state = user_states.get(user_id)
    mode = detect_mode(text)
    print(f"[TEXT_MESSAGE] current_state={state}, detected_mode={mode}")

    if state:
        m = state.get("mode")

        if m in ["ADD", "ASK_NEXT"]:
            print(f"[TEXT_MESSAGE] Routing to add_user_mode")
            return add_user_mode(event, user_id, text)

        if m == "DELETE":
            print(f"[TEXT_MESSAGE] Routing to delete_user_mode")
            return delete_user_mode(event, user_id, text)

    if mode == "ADD":
        print(f"[TEXT_MESSAGE] Detected ADD mode")
        return add_user_mode(event, user_id, text)

    if mode == "DELETE":
        print(f"[TEXT_MESSAGE] Detected DELETE mode")
        return delete_user_mode(event, user_id, text)

    if "เช็คชื่อ" in text:
        print(f"[TEXT_MESSAGE] User requesting face check, waiting for image")
        user_states[user_id] = {"mode": "WAIT_IMAGE"}
        return reply(event, "ส่งรูปมา")

# ================== RUN ==================
if __name__ == "__main__":
    app.run(port=5000, debug=True)