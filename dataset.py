import cv2
import face_recognition
import os
import csv
import json

def build_dataset_from_csv(csv_file="users.csv", user_folder="user", dataset_folder="dataset", max_images=500):
    os.makedirs(dataset_folder, exist_ok=True)
    class_names = []

    if not os.path.exists(csv_file):
        print("❌ ไม่พบ CSV")
        return

    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            name = row[0]
            dataset_path = os.path.join(dataset_folder, name)

            if name not in class_names:
                class_names.append(name)

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

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")

                if face_locations:
                    for (top, right, bottom, left) in face_locations:
                        h, w = frame.shape[:2]
                        bw = right - left
                        bh = bottom - top
                        
                        padding = 0.1
                        
                        x1 = max(0, left - int(bw * padding))
                        y1 = max(0, top - int(bh * padding))
                        x2 = min(w, right + int(bw * padding))
                        y2 = min(h, bottom + int(bh * padding))

                        face = frame[y1:y2, x1:x2]

                        if face.size == 0:
                            continue

                        face = cv2.resize(face, (112, 112))

                        save_file = os.path.join(dataset_path, f"{count}.jpg")
                        cv2.imwrite(save_file, face)

                        count += 1
                        if count >= max_images:
                            break

                for _ in range(2):
                    cap.grab()

                if count >= max_images:
                    break

            cap.release()
            print(f"✅ {name} ได้ {count} รูป")

    # ===== BUILD CLASSES JSON =====
    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=4)

    print("🧠 classes.json ถูกสร้างแล้ว:", class_to_idx)
    print("🎉 สร้าง dataset เสร็จทั้งหมด")

# if __name__ == "__main__":
#     build_dataset_from_csv()