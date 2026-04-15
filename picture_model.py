import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp
from ultralytics import YOLO
import json
import os
from datetime import datetime


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
            nn.Conv2d(3,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*28*28,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        return self.net(x)


model = FaceModel(len(class_to_idx))
model.load_state_dict(torch.load("face_model.pth", map_location="cpu"))
model.eval()


# =========================
# TOOLS
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

yolo = YOLO("yolo11n.pt")

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)


# =========================
# DETECT PERSON (YOLO > 0.6)
# =========================
def detect_persons(frame):
    results = yolo(
        frame,
        conf=0.8,
        classes=[0]
    )[0]

    persons = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != 0 or conf < 0.6:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        persons.append((x1, y1, x2, y2, conf))

    return persons


# =========================
# DETECT BEST FACE
# =========================
def detect_best_face(person_img, conf_threshold=0.8):
    rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(rgb_person)

    if not results_face.detections:
        return None

    ph, pw, _ = person_img.shape

    best_face = None
    best_conf = 0
    best_name = None

    for face_det in results_face.detections:
        bbox = face_det.location_data.relative_bounding_box

        fx = int(bbox.xmin * pw)
        fy = int(bbox.ymin * ph)
        fw = int(bbox.width * pw)
        fh = int(bbox.height * ph)

        fx, fy = max(0, fx), max(0, fy)
        fx2, fy2 = min(pw, fx + fw), min(ph, fy + fh)

        face = person_img[fy:fy2, fx:fx2]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (112,112))
        face_tensor = transform(face_resized).unsqueeze(0)

        with torch.no_grad():
            out = model(face_tensor)
            probs = torch.softmax(out, dim=1)
            conf_score, pred = torch.max(probs, 1)

        conf_score = conf_score.item()
        pred = pred.item()

        if conf_score < conf_threshold:
            continue

        if conf_score > best_conf:
            best_conf = conf_score
            best_face = (fx, fy, fx2, fy2)
            best_name = idx_to_class.get(pred, "Unknown")

    if best_face is None:
        return None

    return {
        "box": best_face,
        "name": best_name,
        "confidence": best_conf
    }


# =========================
# MAIN PROCESS
# =========================
def process_image(image_path, save_dir="day_face_result"):
    os.makedirs(save_dir, exist_ok=True)

    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ อ่านภาพไม่ได้")
        return

    persons = detect_persons(frame)

    detected_results = []

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (x1, y1, x2, y2, p_conf) in enumerate(persons):

        # ===== padding (ช่วยให้ detect หน้าแม่นขึ้น) =====
        pad = 20
        h, w, _ = frame.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        person = frame[y1:y2, x1:x2]

        if person.size == 0:
            continue

        draw_img = person.copy()

        face_data = detect_best_face(person)

        result = {
            "person_id": i,
            "person_confidence": round(p_conf, 4),
            "name": "unknown",
            "face_confidence": 0,
            "image_path": ""
        }

        if face_data:
            name = face_data["name"]
            conf = face_data["confidence"]

            result["name"] = name
            result["face_confidence"] = round(conf, 4)

            fx, fy, fx2, fy2 = face_data["box"]

            cv2.putText(draw_img, f"{name} ({conf:.2f})",
                        (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

            cv2.rectangle(draw_img,
                          (fx, fy),
                          (fx2, fy2),
                          (0,255,0), 2)

        # กรอบคน
        cv2.rectangle(draw_img,
                      (0, 0),
                      (draw_img.shape[1], draw_img.shape[0]),
                      (255,0,0), 2)

        # ===== SAVE =====
        person_path = os.path.join(save_dir, f"{now}_person_{i}.jpg")
        cv2.imwrite(person_path, draw_img)

        result["image_path"] = person_path
        detected_results.append(result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ===== SAVE JSON =====
    json_path = os.path.join(save_dir, f"{now}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detected_results, f, indent=4, ensure_ascii=False)

    print("📊 ผลลัพธ์:", detected_results)

    return detected_results, json_path
