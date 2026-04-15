import cv2
import mediapipe as mp
import os
import csv
import json   # 👈 ต้องมีอันนี้

def build_dataset_from_csv(csv_file="users.csv", user_folder="user", dataset_folder="dataset", max_images=150):
    os.makedirs(dataset_folder, exist_ok=True)

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    class_names = []

    # ===== อ่าน CSV =====
    if not os.path.exists(csv_file):
        print("❌ ไม่พบ CSV")
        return

    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            name = row[0]

            dataset_path = os.path.join(dataset_folder, name)

            # ===== เก็บ class =====
            if name not in class_names:
                class_names.append(name)

            # ===== ถ้ามี dataset แล้ว → ข้าม =====
            if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
                print(f"⏭️ ข้าม {name} (มี dataset แล้ว)")
                continue

            video_path = os.path.join(user_folder, f"{name}.mp4")

            if not os.path.exists(video_path):
                print(f"⚠️ ไม่พบวิดีโอของ {name}")
                continue

            print(f"🎬 กำลังสร้าง dataset: {name}")

            os.makedirs(dataset_path, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

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

                        face = cv2.resize(face, (112, 112))

                        save_file = os.path.join(dataset_path, f"{count}.jpg")
                        cv2.imwrite(save_file, face)

                        count += 1

                        if count >= max_images:
                            break

                if count >= max_images:
                    break

            cap.release()

            print(f"✅ {name} ได้ {count} รูป")

    # =========================
    # 🔥 สร้าง classes.json
    # =========================
    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}

    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=4)

    print("🧠 classes.json ถูกสร้างแล้ว:", class_to_idx)
    print("🎉 สร้าง dataset เสร็จทั้งหมด")

if __name__ == "__main__":
    build_dataset_from_csv()