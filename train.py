import face_recognition
import cv2
import os
import pickle

def train_face_recognition(dataset_path="dataset", encoded_file="trained_faces.pkl"):
    """
    ฟังก์ชันสำหรับเทรนใบหน้าใหม่
    """
    # ===== DELETE OLD MODEL =====
    if os.path.exists(encoded_file):
        os.remove(encoded_file)
        print(f"--- ลบไฟล์โมเดลเก่า ({encoded_file}) เรียบร้อยแล้ว ---")
    
    known_face_encodings = []
    known_face_names = []

    print("--- กำลังเริ่มประมวลผลรูปภาพ (Fresh Training) ---")
    
    # ===== CHECK DATASET FOLDER =====
    if not os.path.exists(dataset_path):
        print(f"❌ Error: ไม่พบโฟลเดอร์ {dataset_path}")
        return

    # ===== PROCESS FACES =====
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        if os.path.isdir(person_dir):
            print(f"กำลังเรียนรู้ใบหน้าของ: {person_name}")
            
            for image_name in os.listdir(person_dir):
                if image_name.startswith('.'):
                    continue
                    
                image_path = os.path.join(person_dir, image_name)
                
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        
                        if len(encodings) > 0:
                            known_face_encodings.append(encodings[0])
                            known_face_names.append(person_name)
                    except Exception as e:
                        print(f"⚠️ ไม่สามารถอ่านรูป {image_name} ได้: {e}")

    # ===== SAVE MODEL =====
    if known_face_encodings:
        with open(encoded_file, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        print(f"--- บันทึกการเรียนรู้ใบหน้าเสร็จสิ้น (รวม {len(known_face_names)} รายการ) ---")
    else:
        print("❌ ไม่พบใบหน้าที่สามารถเทรนได้ กรุณาตรวจสอบรูปภาพใน Dataset")
