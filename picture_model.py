import cv2
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import mediapipe as mp
from ultralytics import YOLO

# ===== Load classes =====
dataset = datasets.ImageFolder("dataset")
classes = dataset.classes

# ===== Model =====
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

model = FaceModel(len(classes))
model.load_state_dict(torch.load("face_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

# ===== YOLO =====
yolo = YOLO("yolo11n_ncnn_model")

# ===== MediaPipe =====
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# ===== อ่านรูป =====
image_path = "1.jpeg"   # 👈 เปลี่ยนตรงนี้
frame = cv2.imread(image_path)

h, w, _ = frame.shape

# ===== YOLO detect =====
# ===== YOLO detect =====
results = yolo(frame)[0]

boxes = results.boxes

for box in boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    if cls != 0:  # person only
        continue

    person = frame[y1:y2, x1:x2]
    if person.size == 0:
        continue

    # ===== detect face =====
    rgb_person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(rgb_person)

    if results_face.detections:
        for face_det in results_face.detections:
            bbox = face_det.location_data.relative_bounding_box

            ph, pw, _ = person.shape

            fx = int(bbox.xmin * pw)
            fy = int(bbox.ymin * ph)
            fw = int(bbox.width * pw)
            fh = int(bbox.height * ph)

            fx, fy = max(0, fx), max(0, fy)
            fx2, fy2 = min(pw, fx + fw), min(ph, fy + fh)

            face = person[fy:fy2, fx:fx2]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (112,112))
            face_tensor = transform(face_resized).unsqueeze(0)

            with torch.no_grad():
                out = model(face_tensor)
                pred = torch.argmax(out,1).item()

            name = classes[pred]

            # ===== draw =====
            cv2.putText(frame, name,
                        (x1 + fx, y1 + fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

            cv2.rectangle(frame,
                          (x1 + fx, y1 + fy),
                          (x1 + fx2, y1 + fy2),
                          (0,255,0), 2)

    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

# ===== แสดงผล =====
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===== save =====
cv2.imwrite("output.jpg", frame)