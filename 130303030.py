import streamlit as st
import cv2
import numpy as np

# STREAMING_CHUNK: 페이지 설정 및 다크 테마 강제 주입...
st.set_page_config(page_title="PI-SHIELD", page_icon="🛡️", layout="wide")

# STREAMING_CHUNK: 하이테크 CSS 스타일 정의...
st.markdown("""
    <style>
        /* 1. 스트림릿 전체 배경을 강제로 딥 네이비 블랙으로 고정 */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0A0D14 !important;
        }
        
        /* 2. 상단 투명 헤더 세팅 */
        header, [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        /* 3. 모든 기본 텍스트 및 레이블 색상을 선명한 흰색으로 강제 전환 */
        h1, h2, h3, h4, h5, h6, p, span, label, li {
            color: #FFFFFF !important;
            font-family: 'Inter', 'Noto Sans KR', sans-serif !important;
        }
        
        /* 4. 체크박스 글씨 색상도 흰색으로 고정 */
        .stCheckbox label span {
            color: #FFFFFF !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
        }
        
        /* 5. 업로드 파일 버튼(Browse files) 스타일 복원 및 네온 블루 커스텀 */
        [data-testid="stFileUploader"] button {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(0, 82, 255, 0.5) !important;
            border-radius: 6px !important;
            font-weight: bold !important;
            padding: 8px 16px !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background-color: #0052FF !important;
            border-color: #00D2FF !important;
        }
        
        /* 6. 사이버틱 대시보드 카드 디자인 */
        .danger-box {
            background: linear-gradient(145deg, #161B26, #0F131C) !important;
            padding: 25px;
            border-radius: 16px;
            border: 1px solid rgba(0, 82, 255, 0.4) !important;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        }
        
        /* 7. 수치 텍스트 컬러 강조 */
        .score-val { 
            font-size: 3rem; 
            font-weight: 800; 
            color: #FF4D4D !important; 
            text-shadow: 0 0 15px rgba(255, 77, 77, 0.6); 
            margin: 10px 0;
        }
        .safe-val { 
            font-size: 2.2rem; 
            font-weight: 800; 
            color: #00D2FF !important; 
            text-shadow: 0 0 15px rgba(0, 210, 255, 0.6); 
        }
        
        /* 8. 일반 버튼 디자인 */
        .stButton>button {
            background: linear-gradient(135deg, #0052FF, #00D2FF) !important;
            color: white !important; 
            border: none !important; 
            font-weight: bold !important; 
            width: 100% !important; 
            border-radius: 8px !important;
            padding: 12px 20px !important;
            box-shadow: 0 4px 15px rgba(0, 82, 255, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5) !important;
        }
    </style>
""", unsafe_allow_html=True)

# STREAMING_CHUNK: 깔끔한 메인 타이틀 생성...
st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 style='color: #FFFFFF !important; font-size: 4rem; font-weight: 900; letter-spacing: -2px; margin-bottom: 0px;'>
            🛡️ PI-<span style='color: #0052FF;'>SHIELD</span>
        </h1>
    </div>
""", unsafe_allow_html=True)

# STREAMING_CHUNK: 파일 업로드 기능 구현...
uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # STREAMING_CHUNK: 데이터 연동 설정...
    detected_info = {
        "명찰": {"detected": True, "base_danger": 25},
        "학교명": {"detected": True, "base_danger": 30},
        "얼굴": {"detected": False, "base_danger": 20},
        "차량번호판": {"detected": True, "base_danger": 20},
        "GPS 정보": {"detected": True, "base_danger": 12}
    }
    
    # STREAMING_CHUNK: 화면 영역 분할...
    col_left, col_right = st.columns([1.2, 1])
    
    # --- 좌측 구역: 사진 Canvas & 범례 ---
    with col_left:
        st.markdown("### 🖼️ 사진 Canvas")
        
        st.markdown("""
            <div style='background-color: #161B26; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.05);'>
                <span style='color: #FF4D4D; font-weight: bold;'>■ 빨간색:</span> 아직 노출됨 &nbsp;&nbsp;&nbsp;&nbsp;
                <span style='color: #00FF66; font-weight: bold;'>■ 초록색:</span> 가리기 선택됨 &nbsp;&nbsp;&nbsp;&nbsp;
                <span style='color: #CC66FF; font-weight: bold;'>■ 보라색:</span> 메타데이터
            </div>
        """, unsafe_allow_html=True)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_column_width=True, caption="[탐지 상자 표시 필드]")
        
        st.button("➕ 영역 직접 추가")

    # STREAMING_CHUNK: 우측 대시보드 렌더링...
    with col_right:
        # 1. 현재 위험도 대시보드
        st.markdown("""
            <div class='danger-box'>
                <p style='margin:0; color:#8F9CAE !important; font-size: 1.1rem; font-weight: 500;'>개인정보 위험도</p>
                <div class='score-val'>87 / 100</div>
                <p style='color:#FF4D4D !important; margin:0; font-weight:bold; font-size: 1.2rem;'>🔥 매우 위험</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. 발견된 정보 리스트 (체크박스)
        st.markdown("### 🔍 발견된 정보 5개")
        
        selected_to_blur = {}
        for item, info in detected_info.items():
            default_val = info["detected"]
            selected_to_blur[item] = st.checkbox(f"   {item}", value=default_val)
            
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        # 3. 실시간 적용 후 위험도 계산 피드백
        current_safe_score = 22
        for item, checked in selected_to_blur.items():
            if not checked and detected_info[item]["detected"]:
                current_safe_score += detected_info[item]["base_danger"]
        
        st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: center; background: #161B26; padding: 18px; border-radius: 12px; border: 1px solid rgba(0, 210, 255, 0.2);'>
                <span style='font-size: 1.1rem; font-weight: bold; color: #FFFFFF !important;'>🛡️ 적용 후 위험도:</span>
                <span class='safe-val'>{min(current_safe_score, 100)}점</span>
            </div>
        """, unsafe_allow_html=True)
        
        # 4. 하단 제어 버튼 셋
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.button("✨ AI 추천")
        with c2: st.button("✅ 모두 선택")
        with c3: st.button("🔄 초기화")
        
        # 5. 최종 결과 생성
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🚀 안전한 사진 생성하기")