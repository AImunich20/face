import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp
from ultralytics import YOLO
import json
import os
from datetime import datetime

detected_results = []
seen_names = set()

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

