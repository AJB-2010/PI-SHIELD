import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
from PIL import Image

st.set_page_config(page_title="PI-SHIELD", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #0A0D14 !important; }
        h1, h2, h3, p, label { color: #FFFFFF !important; font-family: 'Inter', sans-serif !important; }
        .danger-box { background: #161B26; padding: 20px; border-radius: 12px; border: 1px solid #0052FF; text-align: center; }
        .score-val { font-size: 2.5rem; font-weight: 800; color: #FF4D4D; }
        .stButton>button { background: linear-gradient(135deg, #0052FF, #00D2FF); color: white; border: none; border-radius: 6px; width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🛡️ PI-SHIELD</h1>", unsafe_allow_html=True)

def get_face_cascade():
    xml_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_filename):
        github_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try: urllib.request.urlretrieve(github_url, xml_filename)
        except: return None
    try:
        return cv2.CascadeClassifier(xml_filename)
    except:
        return None

def run_real_analysis(cv_img):
    detections = []
    h, w, _ = cv_img.shape
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # 1. 얼굴 감지
    cascade = get_face_cascade()
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, fw, fh) in faces:
            detections.append({"type": "얼굴", "box": [int(x), int(y), int(fw), int(fh)], "danger": 20})
    
    # 2. 텍스트/윤곽선 감지
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 여기서 detected가 0개면 진짜 0개를 반환하도록 로직을 정직하게 수정
    for c in contours:
        x, y, cw, ch = cv2.