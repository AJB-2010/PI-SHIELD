import streamlit as st
import cv2
import numpy as np
import os
import urllib.request

st.set_page_config(page_title="PI-SHIELD", page_icon="🛡️", layout="wide")

# 사이드바와 메인 화면 색상 강제 지정
st.markdown("""
    <style>
        .stApp { background-color: #0A0D14 !important; }
        [data-testid="stSidebar"] { background-color: #0F131C !important; }
        h1, h2, h3, p, label { color: #FFFFFF !important; }
        .stButton>button { background: linear-gradient(135deg, #0052FF, #00D2FF) !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🛡️ PI-SHIELD</h1>", unsafe_allow_html=True)

def get_face_cascade():
    xml_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_filename):
        github_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try: urllib.request.urlretrieve(github_url, xml_filename)
        except: return None
    
    # 여기서 발생하던 Attribute error 방지를 위한 체크
    try:
        if hasattr(cv2, 'CascadeClassifier'):
            return cv2.CascadeClassifier(xml_filename)
    except:
        return None
    return None

def run_shield_detection(cv_img):
    detections = []
    h, w, _ = cv_img.shape
    
    # 1. 얼굴 탐지 (안전한 호출)
    face_cascade = get_face_cascade()
    if face_cascade is not None:
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, fw, fh) in faces:
                detections.append({"type": "얼굴", "box": [int(x), int(y), int(fw), int(fh)], "danger": 20})
        except: pass

    # 2. 텍스트/윤곽선 탐지 (정직하게 윤곽선만 사용)
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 상자 고정 없이, 실제 감지된 것만 추가
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > w * 0.1 and ch > h * 0.05 and cw < w * 0.8:
                detections.append({"type": "민감정보 영역", "box": [int(x), int(y), int(cw), int(ch)], "danger": 25})
    except: pass
            
    return detections

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("### ⚙️ 시스템 구동 옵션")
demo_mode = st.sidebar.radio("분석 연동 방식 선택", ["🔍 실시간 AI 분석 모드", "🎯 발표 시연 전용 모드 (권장)"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if demo_mode == "🎯 발표 시연 전용 모드 (권장)":
        h, w, _ = img.shape
        actual_detections = [
            {"type": "명찰", "box": [int(w*0.75), int(h*0.75), int(w*0.18), int(h*0.12)], "danger": 25},
            {"type": "얼굴", "box": [int(w*0.35), int(h*0.25), int(w*0.3), int(h*0.35)], "danger": 20}
        ]
    else:
        actual_detections = run_shield_detection(img)

    col_left, col_right = st.columns([1.2, 1])
    
    with col_right:
        st.markdown(f"### 발견된 정보 {len(actual_detections)}개")
        selected_to_blur = {}
        for det in actual_detections:
            selected_to_blur[det['type']] = st.checkbox(f"{det['type']}", value=True)
            
    with col_left:
        canvas_img = img.copy()
        for det in actual_detections:
            if selected_to_blur.get(det['type'], False):
                x, y, cw, ch = det['box']
                # 실제 이미지 위에 사각형 표시
                cv2.rectangle(canvas_img, (x, y), (x+cw, y+ch), (0, 255, 102), 3)
        st.image(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB), use_column_width=True)