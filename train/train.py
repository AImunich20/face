import face_recognition
import cv2
import os
import pickle

def train_face_recognition(dataset_path="dataset", encoded_file="trained_faces.pkl"):
    """
    ฟังก์ชันสำหรับเทรนใบหน้าใหม่ โดยจะลบไฟล์โมเดลเก่าทิ้งก่อนเริ่มประมวลผลเสมอ
    """
    # 1. ลบไฟล์เก่าทิ้งก่อน (ถ้ามี) เพื่อให้มั่นใจว่าเป็นการเทรนใหม่ 100%
    if os.path.exists(encoded_file):
        os.remove(encoded_file)
        print(f"--- ลบไฟล์โมเดลเก่า ({encoded_file}) เรียบร้อยแล้ว ---")
    
    known_face_encodings = []
    known_face_names = []

    print("--- กำลังเริ่มประมวลผลรูปภาพ (Fresh Training) ---")
    
    # 2. ตรวจสอบว่ามีโฟลเดอร์ dataset หรือไม่
    if not os.path.exists(dataset_path):
        print(f"❌ Error: ไม่พบโฟลเดอร์ {dataset_path}")
        return

    # 3. วนลูปตามชื่อโฟลเดอร์ใน dataset
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        
        if os.path.isdir(person_dir):
            print(f"กำลังเรียนรู้ใบหน้าของ: {person_name}")
            
            for image_name in os.listdir(person_dir):
                # ข้ามไฟล์แฝงอย่าง .DS_Store (มักเจอใน macOS)
                if image_name.startswith('.'):
                    continue
                    
                image_path = os.path.join(person_dir, image_name)
                
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # โหลดรูป
                        image = face_recognition.load_image_file(image_path)
                        # หา Encoding (Vector 128 มิติ)
                        encodings = face_recognition.face_encodings(image)
                        
                        if len(encodings) > 0:
                            known_face_encodings.append(encodings[0])
                            known_face_names.append(person_name)
                    except Exception as e:
                        print(f"⚠️ ไม่สามารถอ่านรูป {image_name} ได้: {e}")

    # 4. บันทึกค่าที่ Encode แล้วลงไฟล์ใหม่
    if known_face_encodings:
        with open(encoded_file, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        print(f"--- บันทึกการเรียนรู้ใบหน้าเสร็จสิ้น (รวม {len(known_face_names)} รายการ) ---")
    else:
        print("❌ ไม่พบใบหน้าที่สามารถเทรนได้ กรุณาตรวจสอบรูปภาพใน Dataset")

# --- วิธีเรียกใช้งาน ---
# if __name__ == "__main__":
#     train_face_recognition()