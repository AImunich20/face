import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import json
import os
import numpy as np
from datetime import datetime
import time
from PIL import Image

# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD CLASSIFIER (ResNet)
# =========================
checkpoint = torch.load("resnet_face.pth", map_location=DEVICE)
class_names = checkpoint["class_names"]

model_resnet = resnet18(weights=None)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, len(class_names))
model_resnet.load_state_dict(checkpoint["model_state"])
model_resnet = model_resnet.to(DEVICE)
model_resnet.eval()

print("Classes:", class_names)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def classify_face(face_img):
    try:
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model_resnet(x)
            prob = torch.softmax(out, dim=1)
            conf, idx = torch.max(prob, 1)

        return class_names[idx.item()], conf.item()

    except Exception as e:
        print("Classification error:", e)
        return "unknown", 0.0

# -------------------------
# YOLO
# -------------------------
model = YOLO("yolov8n.pt")

model = YOLO("yolov8n.pt")

def get_depth_score(box):
    """
    คนที่อยู่ “ล่างภาพ” = ใกล้กล้อง (front)
    ใช้แค่ตำแหน่ง y เท่านั้น
    """
    x1, y1, x2, y2 = box

    # ใช้กึ่งกลางล่างของคน (stable กว่า y2 ล้วน)
    center_y = (y1 + y2) / 2

    return center_y

def remove_overlap(front_mask, back_mask):
    """
    ลบส่วนที่ซ้อนกับคนหน้าออกจากคนหลัง
    """
    return cv2.bitwise_and(back_mask, cv2.bitwise_not(front_mask))

# =========================
# MEDIAPIPE FACE DETECTION
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img, padding=0.25):
    """
    ใช้ cascade classifier แทน MediaPipe
    return: (x1,y1,x2,y2,confidence)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    detected = []
    for (x, y, w, h) in faces:
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # ขยายกรอบ
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img.shape[1], x2 + pad_x)
        y2 = min(img.shape[0], y2 + pad_y)
        
        detected.append((x1, y1, x2, y2, 0.9))  # confidence = 0.9 (cascade)
    
    return detected

def process(image_path):
    img = cv2.imread(image_path)
    H, W = img.shape[:2]

    # -------------------------
    # เตรียม path + ชื่อไฟล์
    # -------------------------
    os.makedirs("day_face_result", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_image_path = f"day_face_result/{base_name}_{timestamp}.jpg"
    output_json_path = f"day_face_result/{base_name}_{timestamp}.json"

    # -------------------------
    # YOLO
    # -------------------------
    results = model(img)[0]
    CONF_THRES = 0.80

    persons = []
    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r

        if int(cls) == 0 and float(conf) >= CONF_THRES:
            persons.append((int(x1), int(y1), int(x2), int(y2)))

    persons = sorted(persons, key=get_depth_score, reverse=True)

    accumulated_front_mask = np.zeros((H, W), dtype=np.uint8)

    output_data = {
        "image": output_image_path,
        "timestamp": timestamp,
        "persons": []
    }

    # 🔥 เก็บทุก detections (person_id, face_bbox, label, confidence, face_image_path)
    all_detections = []

    for i, (x1, y1, x2, y2) in enumerate(persons):

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        clean_mask = remove_overlap(accumulated_front_mask, mask)
        accumulated_front_mask = cv2.bitwise_or(accumulated_front_mask, clean_mask)

        result = cv2.bitwise_and(img, img, mask=clean_mask)
        person_crop = result[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        faces = detect_faces(person_crop)
        FACE_CONF_THRES = 0.7

        person_info = {
            "person_id": i,
            "bbox": [x1, y1, x2, y2],
            "faces": []
        }

        for j, (fx1, fy1, fx2, fy2, fconf) in enumerate(faces):

            if fconf < FACE_CONF_THRES:
                continue

            face_img = person_crop[fy1:fy2, fx1:fx2]
            if face_img.size == 0:
                continue

            label, conf = classify_face(face_img)

            # map กลับ
            gx1, gy1 = x1 + fx1, y1 + fy1
            gx2, gy2 = x1 + fx2, y1 + fy2

            # save face
            face_filename = f"{base_name}_{timestamp}_p{i}_f{j}_{label}.jpg"
            face_path = os.path.join("day_face_result", face_filename)

            # 🔥 เก็บ detection ทั้งหมด (แม้ confidence ต่ำก็เก็บ)
            all_detections.append({
                "person_id": i,
                "face_bbox": [gx1, gy1, gx2, gy2],
                "label": label,
                "confidence": float(conf),
                "face_image_path": face_path
            })

        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

        if person_info["faces"]:
            output_data["persons"].append(person_info)
        # cv2.imshow("YOLO PERSONS", img)
        # cv2.waitKey(0)

    # -------------------------
    # 🔥 สำหรับแต่ละ label เลือก confidence สูงสุด
    # -------------------------
    label_best = {}  # {label: (detection, confidence)}

    for detection in all_detections:
        label = detection["label"]
        conf = detection["confidence"]

        if label not in label_best or conf > label_best[label][1]:
            label_best[label] = (detection, conf)

    # เรียง ตามความมั่นใจ
    final_detections = sorted(
        [det for det, _ in label_best.values()],
        key=lambda x: x["confidence"],
        reverse=True
    )

    # -------------------------
    # DRAW บนรูป
    # -------------------------
    for det in final_detections:
        gx1, gy1, gx2, gy2 = det["face_bbox"]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (gx1, max(gy1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # -------------------------
    # SAVE FILES
    # -------------------------
    cv2.imwrite(output_image_path, img)

    output_data["persons"] = [
        {
            "person_id": det["person_id"],
            "faces": [
                {
                    "face_bbox": det["face_bbox"],
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "face_image_path": det["face_image_path"]
                }
            ]
        }
        for det in final_detections
    ]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    # -------------------------
    # 🔥 RETURN แค่รายชื่อ (best confidence for each)
    # -------------------------
    return [det["label"] for det in final_detections]


names = process("3.jpg")

print(names)