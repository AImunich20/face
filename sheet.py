import requests
import json

def send_to_google_sheet(json_data):
    """
    ฟังก์ชันสำหรับส่งข้อมูล JSON ไปยัง Google Sheets
    """
    # URL ที่ได้จาก Google Apps Script (ตรวจสอบให้มั่นใจว่าตั้งค่าเป็น Anyone แล้ว)
    url = "https://script.google.com/macros/s/AKfycbz4uXUnIJ2SPGJWduMmf1JTb2gLpAM3EVGp-fe5hY_856grc_ReXF7qCZEsSICVyyuL/exec"
    
    try:
        # ส่งคำขอ POST พร้อมข้อมูล JSON
        response = requests.post(
            url, 
            data=json.dumps(json_data), 
            headers={'Content-Type': 'application/json'},
            timeout=10 # ตั้งเวลา timeout ไว้ 10 วินาทีเผื่อเน็ตช้า
        )
        
        # ตรวจสอบผลลัพธ์
        if response.status_code == 200:
            print(f"✅ สำเร็จ: {response.text}")
            return "Success สามารถบันทึกข้อมูลลง Google Sheets ได้แล้ว กรุณาเช็ขข้อมูลใน Google Sheets ของคุณ"
        else:
            print(f"❌ พลาด: Status Code {response.status_code}")
            print(f"รายละเอียด: {response.text}")
            return f"Error: {response.status_code}"
        
    except Exception as e:
        print(f"❗ เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
        return "Connection Error"

# # ตัวอย่างการอ่านจากไฟล์ .json แล้วส่งเข้าฟังก์ชัน
# with open("day_face_result/U071b6aeedfd9b8e85eb3b847b491e2f6_20260416_153025.json", 'r', encoding='utf-8') as f:
#     local_data = json.load(f)
#     send_to_google_sheet(local_data)