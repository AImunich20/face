import cv2
import mediapipe as mp
import os

name = input("ชื่อคน: ")
save_path = f"dataset/{name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

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

            # กัน out of bounds
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)

            face = frame[y:y2, x:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (112,112))

            cv2.imwrite(f"{save_path}/{count}.jpg", face)
            count += 1

            cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 150:
        break

cap.release()
cv2.destroyAllWindows()