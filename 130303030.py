import streamlit as st
import cv2
import numpy as np

# 1. 페이지 설정 및 테마 CSS (Chic 블랙 & 네온 컬러 유지)
st.set_page_config(page_title="PI-SHIELD", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0A0D14; color: #FFFFFF; }
        .danger-box {
            background: linear-gradient(145deg, #161B26, #0F131C);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(0, 82, 255, 0.2);
            text-align: center;
            margin-bottom: 20px;
        }
        .score-val { font-size: 2.5rem; font-weight: 800; color: #FF4D4D; text-shadow: 0 0 10px rgba(255, 77, 77, 0.5); }
        .safe-val { font-size: 2rem; font-weight: 800; color: #00D2FF; text-shadow: 0 0 10px rgba(0, 210, 255, 0.5); }
        .legend-text { font-size: 0.9rem; margin-bottom: 5px; }
        .stButton>button {
            background: linear-gradient(135deg, #0052FF, #00D2FF);
            color: white; border: none; font-weight: bold; width: 100%; border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# 메인 타이틀
st.markdown("<h2 style='text-align: center; color: #FFFFFF;'>🛡️ PI-SHIELD 결과 화면</h2>", unsafe_allow_html=True)

# 파일 업로더
uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # OpenCV 이미지 변환
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # -------------------------------------------------------------
    # 🛑 [프로그래머 영역] 실제 AI/OpenCV 데이터 연동부
    # 프로그래머는 아래 딕셔너리에 실제 탐지 여부(True/False)와 좌표를 바인딩하면 됩니다.
    # -------------------------------------------------------------
    detected_info = {
        "명찰": {"detected": True, "base_danger": 25},
        "학교명": {"detected": True, "base_danger": 30},
        "얼굴": {"detected": False, "base_danger": 20},
        "차량번호판": {"detected": True, "base_danger": 20},
        "GPS 정보": {"detected": True, "base_danger": 12}
    }
    
    # 좌측 사진 영역과 우측 대시보드 영역 분할 (기획서 비율 반영)
    col_left, col_right = st.columns([1.2, 1])
    
    # --- 좌측 구역: 사진 Canvas & 범례 ---
    with col_left:
        st.markdown("### 🖼️ 사진 Canvas")
        
        # 범례 표시
        st.markdown("""
            <div style='background-color: #161B26; padding: 12px; border-radius: 8px; margin-bottom: 15px;'>
                <span style='color: #FF4D4D; font-weight: bold;'>■ 빨간색:</span> 아직 노출됨 &nbsp;&nbsp;&nbsp;&nbsp;
                <span style='color: #00FF66; font-weight: bold;'>■ 초록색:</span> 가리기 선택됨 &nbsp;&nbsp;&nbsp;&nbsp;
                <span style='color: #CC66FF; font-weight: bold;'>■ 보라색:</span> 메타데이터
            </div>
        """, unsafe_allow_html=True)
        
        # [탐지 상자 표시] 
        # 프로그래머가 체크박스 상태에 따라 이미지 위에 cv2.rectangle로 사각형을 그릴 영역입니다.
        # 시연을 위해 우선 원본을 띄웁니다.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_column_width=True, caption="[탐지 상자 표시 필드]")
        
        # 영역 직접 추가 버튼
        st.button("➕ 영역 직접 추가")

    # --- 우측 구역: 위험도 점수 & 발견된 정보 리스트 ---
    with col_right:
        # 1. 현재 위험도 대시보드
        st.markdown("<div class='danger-box'>", unsafe_allow_html=True)
        st.markdown("<p style='margin:0; color:#8F9CAE;'>개인정보 위험도</p>", unsafe_allow_html=True)
        st.markdown("<div class='score-val'>87 / 100</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#FF4D4D; margin:0; font-weight:bold;'>🔥 매우 위험</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 2. 발견된 정보 리스트 (체크박스 인터랙션)
        st.markdown("### 🔍 발견된 정보 5개")
        
        selected_to_blur = {}
        
        # 기획서에 있는 체크박스 리스트 구성 및 초기값 설정
        for item, info in detected_info.items():
            # 얼굴을 제외하고 나머지는 기획서대로 기본 체크(True) 상태로 둠
            default_val = info["detected"]
            
            # 체크박스 생성
            selected_to_blur[item] = st.checkbox(f"{item}", value=default_val)
            
        st.markdown("---")
        
        # 3. 적용 후 위험도 연동계산 (인터랙션 피드백)
        # 체크를 해제할수록(안 가릴수록) 위험도가 실시간으로 올라가는 효과를 줍니다.
        current_safe_score = 22
        for item, checked in selected_to_blur.items():
            if not checked and detected_info[item]["detected"]:
                current_safe_score += detected_info[item]["base_danger"]
        
        st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: center; background: #161B26; padding: 15px; border-radius: 8px;'>
                <span style='font-size: 1.1rem; font-weight: bold;'>🛡️ 적용 후 위험도:</span>
                <span class='safe-val'>{min(current_safe_score, 100)}점</span>
            </div>
        """, unsafe_allow_html=True)
        
        # 4. 제어용 버튼 셋 (AI 추천 적용, 모두 선택, 초기화)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.button("✨ AI 추천 적용")
        with c2: st.button("✅ 모두 선택")
        with c3: st.button("🔄 초기화")
        
        # 5. 최종 결과 생성 버튼
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🚀 안전한 사진 생성하기")