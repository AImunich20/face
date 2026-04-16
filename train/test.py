import face_recognition
import cv2
import pickle
import os
import sys

# --- การตั้งค่า (Settings) ---
encoded_file = "trained_faces.pkl"  # ไฟล์ข้อมูลใบหน้าที่เทรนไว้
test_image_path = "22.jpg"  # ระบุ Path ของรูปที่ต้องการทดสอบที่นี่
output_path = "result.jpg"     # Path สำหรับบันทึกรูปผลลัพธ์
# -----------------------------

def main():
    # 1. ตรวจสอบไฟล์โมเดล
    if not os.path.exists(encoded_file):
        print(f"❌ ไม่พบไฟล์ {encoded_file}")
        print("กรุณารันโค้ด Train ก่อนเพื่อสร้างไฟล์นี้ครับ")
        sys.exit(1)

    # 2. ตรวจสอบไฟล์รูปภาพทดสอบ
    if not os.path.exists(test_image_path):
        print(f"❌ ไม่พบไฟล์รูปภาพทดสอบ: {test_image_path}")
        sys.exit(1)

    # 3. โหลดข้อมูลใบหน้าที่เทรนไว้แล้ว (Encodings)
    print(f"➕ กำลังโหลดข้อมูลใบหน้าที่เทรนไว้จาก {encoded_file}...")
    with open(encoded_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"✅ โหลดสำเร็จ! รู้จักทั้งหมด {len(set(known_face_names))} คน ({len(known_face_encodings)} รูป)")

    # 4. โหลดรูปภาพทดสอบและค้นหาใบหน้า
    print(f"📸 กำลังประมวลผลรูปภาพ: {test_image_path}...")
    
    # โหลดรูปภาพสำหรับการค้นหา (dlib format)
    image_to_test = face_recognition.load_image_file(test_image_path)
    
    # โหลดรูปภาพสำหรับการวาดผลลัพธ์ (OpenCV format)
    image_to_draw = cv2.imread(test_image_path)
    
    # หากรูปภาพทดสอบใหญ่เกินไป ให้ย่อขนาดลงเพื่อความรวดเร็ว (เลือกเปิดใช้ถ้าจำเป็น)
    # scale_percent = 50 
    # width = int(image_to_draw.shape[1] * scale_percent / 100)
    # height = int(image_to_draw.shape[0] * scale_percent / 100)
    # image_to_draw = cv2.resize(image_to_draw, (width, height))
    # image_to_test = cv2.resize(image_to_test, (width, height))


    # ค้นหาตำแหน่งใบหน้าทั้งหมดในรูป
    face_locations = face_recognition.face_locations(image_to_test)
    # แปลงใบหน้าเป็น Encoding (ตัวเลข)
    face_encodings = face_recognition.face_encodings(image_to_test, face_locations)

    print(f"📍 พบใบหน้าทั้งหมด {len(face_locations)} ใบหน้า")

    # 5. เปรียบเทียบใบหน้า
    # วนลูปผ่านใบหน้าแต่ละใบหน้าที่พบในรูปทดสอบ
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        # ตรวจสอบว่าใบหน้าตรงกับฐานข้อมูลไหม
        # Tolerance ยิ่งต่ำยิ่งเข้มงวด (Default คือ 0.6)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown (ไม่รู้จัก)"

        # ใช้วิธีหาระยะห่าง (Distance) เพื่อหาใบหน้าที่ใกล้เคียงที่สุด
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin() # หาค่าที่น้อยที่สุด (ใกล้เคียงที่สุด)
            
            # ถ้าค่าระยะห่างน้อยพอ และตรงกับที่ matches บอกว่าใช่
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # --- แก้ไขจากของเดิมเป็นแบบนี้ ---
        min_dist = min(face_distances) if len(face_distances) > 0 else 0
        
        print(f"🔍 ผลการวิเคราะห์: {name} (Distance: {min_dist:.4f})")
        # -----------------------------

        # 6. วาดผลลัพธ์บนรูปภาพ
        # วาดกรอบสี่เหลี่ยมรอบใบหน้า (สีเขียว, ความหนา 2)
        cv2.rectangle(image_to_draw, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # วาดแถบสีด้านล่างสำหรับใส่ชื่อ (เพื่อให้เห็นตัวหนังสือชัด)
        cv2.rectangle(image_to_draw, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        
        # เขียนชื่อ (ใช้ Font OpenCV พื้นฐาน ไม่รองรับภาษาไทย)
        # หากต้องการแสดงภาษาไทยบนรูป ต้องใช้ library เพิ่มเติมเช่น Pillow
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_to_draw, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # 7. แสดงผลและบันทึกรูป
    # บันทึกรูปผลลัพธ์
    cv2.imwrite(output_path, image_to_draw)
    print(f"💾 บันทึกรูปผลลัพธ์ที่: {output_path}")

    # แสดงรูปภาพ (หากรันบนเครื่องที่มีหน้าจอ)
    try:
        cv2.imshow('Face Recognition Test', image_to_draw)
        print("⌨️  กดปุ่มใดก็ได้บนคีย์บอร์ดเพื่อปิดหน้าต่างรูปภาพ...")
        cv2.waitKey(0) # รอจนกว่าจะกดปุ่ม
        cv2.destroyAllWindows()
    except Exception:
        print("🖥️  ไม่สามารถแสดงหน้าต่างรูปภาพได้ (อาจรันแบบไม่มีหน้าจอ) โปรดดูไฟล์ผลลัพธ์ที่บันทึกไว้แทน")

if __name__ == "__main__":
    main()