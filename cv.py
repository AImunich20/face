import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from retinaface import RetinaFace
import json

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
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =========================
# YOLO PERSON
# =========================
yolo = YOLO("yolo11n.pt")

def detect_persons(frame):
    results = yolo(frame, conf=0.6, classes=[0])[0]
    persons = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        persons.append((x1, y1, x2, y2, conf))

    return persons

# =========================
# RETINAFACE
# =========================
def detect_faces(frame):
    faces = RetinaFace.detect_faces(frame)
    results = []

    if isinstance(faces, dict):
        for k in faces:
            face = faces[k]
            x1, y1, x2, y2 = face["facial_area"]
            conf = face["score"]

            if conf > 0.7:
                results.append((x1, y1, x2, y2, conf))

    return results

# =========================
# SELECT BEST FACE
# =========================
def select_best_face(faces):
    if len(faces) == 0:
        return None

    best_face = None
    best_score = 0

    for (x1, y1, x2, y2, conf) in faces:
        area = (x2 - x1) * (y2 - y1)
        score = conf * area

        if score > best_score:
            best_score = score
            best_face = (x1, y1, x2, y2, conf)

    return best_face

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)

    inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)

    box1_area = (x2-x1)*(y2-y1)
    box2_area = (x2b-x1b)*(y2b-y1b)

    union = box1_area + box2_area - inter_area

    if union == 0:
        return 0

    return inter_area / union

def filter_front_persons(persons, iou_thresh=0.4):
    filtered = []

    for p in persons:
        px1, py1, px2, py2, _ = p

        keep = True
        for fp in filtered:
            fx1, fy1, fx2, fy2, _ = fp

            iou = compute_iou(
                (px1, py1, px2, py2),
                (fx1, fy1, fx2, fy2)
            )

            if iou > iou_thresh:
                keep = False
                break

        if keep:
            filtered.append(p)

    return filtered

# =========================
# PROCESS IMAGE
# =========================
def process_image(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        print("โหลดภาพไม่ได้")
        return

    persons = detect_persons(frame)

    # 🔥 เรียงล่าง → บน
    persons = sorted(persons, key=lambda x: x[3], reverse=True)

    # 🔥 เอาเฉพาะคนหน้า
    persons = filter_front_persons(persons)

    for (px1, py1, px2, py2, _) in persons:
        person_crop = frame[py1:py2, px1:px2]

        faces = detect_faces(person_crop)
        best_face = select_best_face(faces)

        if best_face is None:
            continue

        fx1, fy1, fx2, fy2, fconf = best_face

        face_img = person_crop[fy1:fy2, fx1:fx2]

        if face_img.shape[0] < 50 or face_img.shape[1] < 50:
            continue

        face_img = cv2.resize(face_img, (112,112))
        face_tensor = transform(face_img).unsqueeze(0)

        with torch.no_grad():
            output = model(face_tensor)
            pred = torch.argmax(output, dim=1).item()
            name = idx_to_class[pred]

        print("Detected:", name)

        # draw
        cv2.rectangle(frame, (px1+fx1, py1+fy1), (px1+fx2, py1+fy2), (0,255,0), 2)
        cv2.putText(frame, name, (px1+fx1, py1+fy1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# RUN
# =========================
process_image("1.jpg")