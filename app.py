import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from deepface import DeepFace

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Seminar Satisfaction Report",
    page_icon="📊",
    layout="wide"
)

# --- 1. โหลดโมเดล PyTorch ที่เทรนเอง ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@st.cache_resource
def load_custom_model():
    model = models.mobilenet_v3_small()
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
    
    # โหลด Weights ที่เทรนได้
    model.load_state_dict(torch.load("custom_emotion_model.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

try:
    custom_model = load_custom_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดล custom_emotion_model.pth ได้: {e}")

# Transform สำหรับแปลงภาพใบหน้าก่อนส่งเข้าโมเดล
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- น้ำหนักคะแนนอารมณ์ ---
EMOTION_WEIGHTS = {
    'happy': 1.0,
    'surprise': 0.8,
    'neutral': 0.6,
    'sad': 0.3,
    'fear': 0.2,
    'angry': 0.0,
    'disgust': 0.0
}

EMOTION_COLORS = {
    'happy': (0, 255, 0),       # Green
    'surprise': (0, 255, 255),  # Yellow
    'neutral': (255, 200, 0),   # Cyan/Light Blue
    'sad': (0, 165, 255),      # Orange
    'fear': (0, 100, 255),     # Dark Orange
    'angry': (0, 0, 255),      # Red
    'disgust': (128, 0, 128)   # Purple
}

# --- Header ส่วนหัวเว็บ ---
st.title("📊 ระบบรายงานประเมินความพึงพอใจงานสัมมนา (Custom Trained Model)")
st.markdown("อัปโหลดภาพถ่ายบรรยากาศการสัมมนาเพื่อวิเคราะห์อารมณ์ด้วย AI Model ที่เทรนขึ้นเอง")
st.divider()

# --- Sidebar สำหรับอัปโหลดไฟล์ ---
st.sidebar.header("⚙️ อัปโหลดข้อมูล")
uploaded_file = st.sidebar.file_uploader("เลือกรูปภาพสัมมนา (.jpg, .png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    orig_image = image.copy()
    
    with st.spinner('กำลังตรวจจับใบหน้าและวิเคราะห์อารมณ์ด้วยโมเดล custom...'):
        try:
            # 1. ใช้ DeepFace (MTCNN) ในการตรวจจับตำแหน่งใบหน้า (Face Detection)
            face_results = DeepFace.extract_faces(
                image, 
                detector_backend='mtcnn', 
                enforce_detection=False
            )
            
            emotions_count = {e: 0 for e in EMOTION_WEIGHTS.keys()}
            total_score = 0
            valid_faces = 0

            for res in face_results:
                confidence = res.get('confidence', 0)
                if confidence < 0.8: # ข้ามบริเวณที่จับใบหน้าได้ไม่ชัดเจน
                    continue

                facial_area = res.get('facial_area', {})
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 0)
                h = facial_area.get('h', 0)

                # ครอบตัด (Crop) ใบหน้า
                face_crop = image[max(0, y):y+h, max(0, x):x+w]
                if face_crop.size == 0:
                    continue

                # แปลง BGR เป็น RGB และแปลงเป็น PIL Image
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)
                
                # 2. ส่งใบหน้าเข้า Custom PyTorch Model เพื่อทำนายอารมณ์
                tensor_img = img_transform(pil_img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = custom_model(tensor_img)
                    _, preds = torch.max(outputs, 1)
                    dominant_e = CLASS_NAMES[preds[0]]

                if dominant_e in EMOTION_WEIGHTS:
                    emotions_count[dominant_e] += 1
                    total_score += EMOTION_WEIGHTS[dominant_e]
                    valid_faces += 1
                    
                    color = EMOTION_COLORS.get(dominant_e, (255, 255, 255))
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, dominant_e, (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # คำนวณคะแนนเฉลี่ย (%)
            avg_satisfaction = (total_score / valid_faces * 100) if valid_faces > 0 else 0
            dominant_overall = max(emotions_count, key=emotions_count.get) if valid_faces > 0 else "N/A"

            # --- Metrics ---
            col1, col2, col3 = st.columns(3)
            col1.metric("👥 ผู้เข้าฟังที่พบ (คน)", f"{valid_faces}")
            col2.metric("⭐ คะแนนความพึงพอใจเฉลี่ย", f"{avg_satisfaction:.1f}%")
            col3.metric("🎭 อารมณ์หลักของห้อง", dominant_overall.upper())

            st.divider()

            # --- Visual Comparison ---
            st.subheader("🖼️ ผลการตรวจจับและวิเคราะห์ใบหน้า")
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.image(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB), caption="รูปภาพต้นฉบับ", use_container_width=True)
            with img_col2:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="ผลวิเคราะห์จาก Custom Model", use_container_width=True)

            st.divider()

            # --- Charts ---
            st.subheader("📈 สรุปสัดส่วนอารมณ์ของผู้เข้าฟัง")
            df_emotions = pd.DataFrame({
                'Emotion': list(emotions_count.keys()),
                'Count': list(emotions_count.values())
            })
            df_emotions = df_emotions[df_emotions['Count'] > 0]

            if not df_emotions.empty:
                graph_col1, graph_col2 = st.columns(2)
                with graph_col1:
                    fig_pie = px.pie(df_emotions, values='Count', names='Emotion', title="สัดส่วนเปอร์เซ็นต์อารมณ์", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with graph_col2:
                    fig_bar = px.bar(df_emotions, x='Emotion', y='Count', color='Emotion', title="จำนวนผู้เข้าฟังแบ่งตามอารมณ์")
                    st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
else:
    st.info("👈 กรุณาอัปโหลดรูปภาพสัมมนาทางแถบเมนูด้านซ้ายเพื่อเริ่มการวิเคราะห์")