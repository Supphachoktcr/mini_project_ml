import cv2
from deepface import DeepFace

# 1. กำหนด Weight คะแนนสำหรับบรรยากาศการฟังสัมมนา (0.0 ถึง 1.0)
EMOTION_WEIGHTS = {
    'happy': 1.0,      # พึงพอใจมาก / ยิ้ม
    'surprise': 0.85,  # สนใจ / ตื่นเต้น
    'neutral': 0.65,   # นั่งฟังปกติ (ตั้งใจฟัง)
    'sad': 0.35,       # อาจจะเหนื่อยหรือก้มมองเอกสาร
    'fear': 0.25,
    'disgust': 0.15,
    'angry': 0.10      # หงุดหงิด / ไม่พอใจ
}

# กำหนดสีของกรอบ (BGR Format)
EMOTION_COLORS = {
    'happy': (0, 255, 0),       # เขียว
    'surprise': (0, 255, 255),  # เหลือง
    'neutral': (255, 200, 0),   # ฟ้า/เขียวอ่อน
    'sad': (0, 165, 255),       # ส้ม
    'fear': (0, 100, 255),      # ส้มเข้ม
    'disgust': (0, 0, 255),     # แดง
    'angry': (0, 0, 255)        # แดง
}

# 2. โหลดรูปภาพ
image_path = "test_person.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: หาไฟล์รูป {image_path} ไม่เจอ กรุณาเช็กชื่อไฟล์อีกครั้ง")
else:
    print("กำลังวิเคราะห์อารมณ์ใบหน้าคนในภาพสัมมนา...")
    
    # วิเคราะห์อารมณ์ด้วย detector_backend='mtcnn'
    results = DeepFace.analyze(
        img_path=image_path, 
        actions=['emotion'], 
        detector_backend='mtcnn',
        enforce_detection=False
    )
    
    total_faces = len(results)
    total_satisfaction_score = 0.0

    # 3. วนลูปประมวลผลใบหน้าที่ตรวจพบ
    for idx, result in enumerate(results, 1):
        x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
        dominant_emotion = result['dominant_emotion']
        confidence = result['emotion'][dominant_emotion]
        
        # คำนวณคะแนนพึงพอใจของรายบุคคล
        face_weight = EMOTION_WEIGHTS.get(dominant_emotion, 0.5)
        total_satisfaction_score += face_weight
        
        # เลือกลักษณะสีของกรอบตามอารมณ์
        color = EMOTION_COLORS.get(dominant_emotion, (0, 255, 0))
        
        # พิมพ์บอกรายละเอียดใน Terminal
        print(f"คนเรื่องที่ {idx}: อารมณ์={dominant_emotion.upper()} ({confidence:.1f}%) | คะแนนพึงพอใจ={face_weight*100:.0f}%")
        
        # วาดกรอบสี่เหลี่ยม + ป้ายชื่ออารมณ์
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{dominant_emotion} ({confidence:.0f}%)"
        cv2.putText(img, label, (x, max(y - 10, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. คำนวณคะแนนพึงพอใจเฉลี่ยรวมทั้งสัมมนา (Overall Satisfaction Score)
    if total_faces > 0:
        avg_satisfaction = (total_satisfaction_score / total_faces) * 100
    else:
        avg_satisfaction = 0.0

    print("\n==========================================")
    print(f"จำนวนผู้เข้าร่วมที่ตรวจพบทั้งหมด: {total_faces} คน")
    print(f"คะแนนความพึงพอใจเฉลี่ยสัมมนา: {avg_satisfaction:.1f}%")
    print("==========================================\n")

    # แปะป้ายคะแนนรวมมุมบนซ้ายของภาพ
    summary_text = f"Audience: {total_faces} | Satisfaction: {avg_satisfaction:.1f}%"
    cv2.rectangle(img, (10, 10), (420, 55), (0, 0, 0), -1)  # แถบพื้นหลังดำ
    cv2.putText(img, summary_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 5. แสดงภาพผลลัพธ์
    cv2.imshow("Seminar Satisfaction Analysis", img)
    print("กดปุ่มใดก็ได้บนแป้นพิมพ์เพื่อปิดหน้าต่างรูปภาพ")
    cv2.waitKey(0)
    cv2.destroyAllWindows()