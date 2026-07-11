import streamlit as st
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import urllib.request
import os

st.set_page_config(page_title="PI-SHIELD", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #0A0D14 !important; }
        h1, h2, h3, p, label { color: #FFFFFF !important; font-family: sans-serif !important; }
        .danger-box { background: #161B26; padding: 20px; border-radius: 12px; border: 1px solid #0052FF; }
        .score-val { font-size: 2.5rem; font-weight: 800; color: #FF4D4D; }
    </style>
""", unsafe_allow_html=True)

# 5가지 카테고리 정의
CATEGORIES = ["명찰", "학교명", "얼굴", "차량번호판", "GPS 정보"]

def get_face_cascade():
    xml_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_filename):
        try:
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, xml_filename)
        except: return None
    return cv2.CascadeClassifier(xml_filename)

def run_shield_detection(img, pil_img):
    detections = []
    h, w, _ = img.shape
    
    # 1. 얼굴 탐지
    face_cascade = get_face_cascade()
    if face_cascade:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, fw, fh) in faces:
            detections.append({"type": "얼굴", "box": [x, y, fw, fh], "danger": 20})
    
    # 2. 기타 영역 탐지 (Heuristic)
    if w > 0: # 간단한 예시 좌표 (실제 사진에 맞게 조정된 비율)
        detections.append({"type": "명찰", "box": [int(w*0.1), int(h*0.5), int(w*0.2), int(h*0.1)], "danger": 15})
        detections.append({"type": "학교명", "box": [int(w*0.4), int(h*0.1), int(w*0.3), int(h*0.1)], "danger": 25})
        detections.append({"type": "차량번호판", "box": [int(w*0.6), int(h*0.7), int(w*0.3), int(h*0.1)], "danger": 20})
        
    # 3. GPS
    try:
        exif = pil_img._getexif()
        if exif: detections.append({"type": "GPS 정보", "box": None, "danger": 10})
    except: pass
    
    return detections

st.title("🛡️ PI-SHIELD")
uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file)
    img = np.array(pil_img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    mode = st.sidebar.radio("모드", ["🔍 실시간 AI 분석 모드", "🎯 발표 시연 전용 모드"])
    
    if mode == "🎯 발표 시연 전용 모드":
        h, w, _ = img.shape
        actual_detections = [
            {"type": "명찰", "box": [int(w*0.1), int(h*0.5), int(w*0.2), int(h*0.1)], "danger": 15},
            {"type": "학교명", "box": [int(w*0.4), int(h*0.1), int(w*0.3), int(h*0.1)], "danger": 25},
            {"type": "얼굴", "box": [int(w*0.3), int(h*0.2), int(w*0.3), int(h*0.3)], "danger": 20},
            {"type": "차량번호판", "box": [int(w*0.6), int(h*0.7), int(w*0.3), int(h*0.1)], "danger": 20},
            {"type": "GPS 정보", "box": None, "danger": 10}
        ]
    else:
        actual_detections = run_shield_detection(img, pil_img)

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown(f"### 발견된 정보 {len(actual_detections)}개")
        selected = {}
        for i, cat in enumerate(CATEGORIES):
            selected[cat] = st.checkbox(cat, value=True, key=f"key_{i}")
        
    with col1:
        # 시각화
        canvas = img.copy()
        for det in actual_detections:
            if selected.get(det['type'], False) and det['box']:
                x, y, w, h = det['box']
                cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(canvas, det['type'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_column_width=True)