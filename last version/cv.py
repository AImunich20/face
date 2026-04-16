import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import mediapipe as mp
import json
import os
import numpy as np
from datetime import datetime
import time

# =========================
# FIX PROTOBUF
# =========================
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

DEBUG = True

# =========================
# LOAD LABEL
# =========================
with open("classes.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# =========================
# MODEL
# =========================
class FaceModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# FACE CLASSIFICATION MODEL
# -------------------------
face_model = FaceModel(len(class_to_idx))
face_model.load_state_dict(torch.load("face_model.pth", map_location="cpu"))
face_model.eval()

# -------------------------
# YOLO
# -------------------------
yolo_model = YOLO("yolov8n.pt")

def classify_face(face_img):
    try:
        face_resized = cv2.resize(face_img, (112, 112))
        tensor = transform(face_resized).unsqueeze(0)

        with torch.no_grad():
            out = face_model(tensor)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)

        label = idx_to_class[pred.item()]
        confidence = conf.item()

        return label, confidence

    except Exception as e:
        print("Classification error:", e)
        return "unknown", 0.0
    
# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

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
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,  # 0 = ใกล้, 1 = ไกล
    min_detection_confidence=0.5
)

def detect_faces(img, padding=0.25):
    """
    padding = 0.25 → ขยาย 25% ของขนาดหน้า
    return: (x1,y1,x2,y2,confidence)
    """
    h, w = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    faces = []

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            conf = det.score[0]

            # -------------------------
            # 🔥 ขยายกรอบ
            # -------------------------
            bw = x2 - x1
            bh = y2 - y1

            pad_x = int(bw * padding)
            pad_y = int(bh * padding)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            faces.append((x1, y1, x2, y2, conf))

    return faces

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
        cv2.imshow("YOLO PERSONS", img)
        cv2.waitKey(0)

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