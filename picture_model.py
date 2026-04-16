import cv2
import face_recognition
import pickle
import numpy as np
import os
import json
from datetime import datetime
from ultralytics import YOLO

# =========================
# CONFIG & LOAD DATASET
# =========================
ENCODED_FILE = "trained_faces.pkl"
YOLO_MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.80
CONF_PEOPLE_THRES = 0.50
FACE_TOLERANCE = 0.5

if not os.path.exists(ENCODED_FILE):
    print("❌ ไม่พบไฟล์ trained_faces.pkl กรุณาเทรนก่อนครับ")
    exit()

with open(ENCODED_FILE, "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

yolo_model = YOLO(YOLO_MODEL_PATH)

def get_depth_score(box):
    x1, y1, x2, y2 = box
    return (y1 + y2) / 2

def remove_overlap(front_mask, back_mask):
    return cv2.bitwise_and(back_mask, cv2.bitwise_not(front_mask))

# =========================
# PROCESS FUNCTION (WITH JSON SAVE)
# =========================
def process(image_path):
    img = cv2.imread(image_path)
    if img is None: return []
    H, W = img.shape[:2]
    
    os.makedirs("day_face_result", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_image_path = f"day_face_result/{base_name}_{timestamp}.jpg"
    output_json_path = f"day_face_result/{base_name}_{timestamp}.json"

    results = yolo_model(img)[0]
    persons = []
    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r
        if int(cls) == 0 and float(conf) >= CONF_PEOPLE_THRES:
            persons.append((int(x1), int(y1), int(x2), int(y2)))

    persons = sorted(persons, key=get_depth_score, reverse=True)
    accumulated_front_mask = np.zeros((H, W), dtype=np.uint8)
    all_detections = []

    for i, (x1, y1, x2, y2) in enumerate(persons):
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        clean_mask = remove_overlap(accumulated_front_mask, mask)
        accumulated_front_mask = cv2.bitwise_or(accumulated_front_mask, clean_mask)

        person_crop = cv2.bitwise_and(img, img, mask=clean_mask)[y1:y2, x1:x2]
        if person_crop.size == 0: continue

        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_crop, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)

        for j, (face_encoding, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=FACE_TOLERANCE)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                conf_score = 1 - face_distances[best_match_index]
            else:
                conf_score = 0

            gx1, gy1 = x1 + left, y1 + top
            gx2, gy2 = x1 + right, y1 + bottom

            all_detections.append({
                "person_id": i,
                "face_bbox": [int(gx1), int(gy1), int(gx2), int(gy2)],
                "label": name,
                "confidence": float(conf_score)
            })

    # ===== SELECT BEST CANDIDATE =====
    label_best = {}
    for det in all_detections:
        label = det["label"]
        if label == "Unknown": continue 
        if label not in label_best or det["confidence"] > label_best[label]["confidence"]:
            label_best[label] = det

    final_detections = list(label_best.values())

    # ===== DRAW ON IMAGE =====
    display_names = []
    for idx, det in enumerate(final_detections, start=1):
        b = det["face_bbox"]
        label = det["label"]
        confidence = det["confidence"]
        
        display_text = f"{idx}. {label}"
        text = f"{idx}."
        display_names.append(display_text)
        
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        
        cv2.putText(
            img, 
            f"{text} ({confidence:.2f})", 
            (b[0], max(b[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )

    # ===== SAVE FILES =====
    cv2.imwrite(output_image_path, img)

    output_data = {
        "source_image": image_path,
        "processed_at": timestamp,
        "results": final_detections,
        "display_list": display_names
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"✅ บันทึกรูปภาพ: {output_image_path}")
    print(f"✅ บันทึก JSON: {output_json_path}")

    return display_names, output_image_path, output_json_path