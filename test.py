import cv2
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import mediapipe as mp

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

cap = cv2.VideoCapture(0)

# ===== MediaPipe =====
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)

            face = frame[y:y2, x:x2]

            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (112,112))
            face_tensor = transform(face_resized).unsqueeze(0)

            with torch.no_grad():
                out = model(face_tensor)
                pred = torch.argmax(out,1).item()

            name = classes[pred]

            cv2.putText(frame, name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0),2)

            cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()