import streamlit as st
import cv2
import numpy as np

# 1. 페이지 기본 설정 및 Chic 블랙&블루 테마 CSS 주입
st.set_page_config(page_title="PI-SHIELD", page_icon="🛡️", layout="centered")

st.markdown("""
    <style>
        /* 딥 네이비 블랙 배경 */
        .main { 
            background-color: #0A0D14; 
            color: #FFFFFF; 
        }
        /* 일렉트릭 블루 그라디언트 버튼 */
        .stButton>button {
            background: linear-gradient(135deg, #0052FF, #00D2FF);
            color: white;
            border: none;
            padding: 12px 24px;
            font-weight: bold;
            font-size: 1.1rem;
            border-radius: 8px;
            width: 100%;
            box-shadow: 0 4px 15px rgba(0, 82, 255, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
            transform: translateY(-2px);
        }
        /* 사이버 대시보드 카드 */
        .danger-card {
            background: linear-gradient(145deg, #161B26, #0F131C);
            padding: 25px;
            border-radius: 16px;
            border: 1px solid rgba(0, 82, 255, 0.2);
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        /* 네온 블루 위험도 점수 */
        .score-text {
            font-size: 3.5rem;
            font-weight: 800;
            color: #00D2FF;
            text-shadow: 0 0 15px rgba(0, 210, 255, 0.6);
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# 2. 메인 타이틀 (수정완료: PI-SHIELD)
st.markdown("""
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1 style='color: #FFFFFF; font-size: 3rem; font-weight: 900; letter-spacing: -1px;'>
            🛡️ PI-<span style='color: #0052FF;'>SHIELD</span>
        </h1>
        <p style='color: #6C7A9C; font-size: 1.1rem; font-weight: 500;'>
            얼굴은 선명하게, 개인정보는 철저하게. 디지털 시민의 프라이버시 방패
        </p>
    </div>
""", unsafe_allow_html=True)

# 3. 파일 업로더
uploaded_file = st.file_uploader("보호할 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

# 4. 가상의 결과 출력 화면 (프로그래머가 실제 OpenCV 엔진을 연결할 자리)
if uploaded_file is not None:
    # 이미지를 OpenCV 포맷으로 변환
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 임시 가상 로직 (실제 개발 시 개발팀이 짠 함수로 교체될 예정)
    mock_danger_score = 88 
    
    st.markdown("---")
    
    # 위험도 출력 카드
    st.markdown(f"""
        <div class="danger-card">
            <h3 style='margin: 0; color: #8F9CAE; font-size: 1rem; letter-spacing: 0.5px;'>
                SPECIFICITY EXPOSURE RISK (특정성 노출 위험도)
            </h3>
            <div class="score-text">{mock_danger_score}%</div>
            <p style='color: #FF4B4B; font-size: 0.95rem; font-weight: 600; margin-top: 5px;'>
                ⚠️ 주변 글자 정보 및 외곽 배경 패턴 감지! 시스템이 즉시 블러를 가동합니다.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 결과 비교 화면 레이아웃 (좌: 원본, 우: 결과)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p style='text-align:center; color:#6C7A9C;'>[Original Image]</p>", unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True)
        
    with col2:
        st.markdown("<p style='text-align:center; color:#00D2FF; font-weight:bold;'>[PI-SHIELD Blur Protection]</p>", unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True)
        
    # 다운로드 버튼
    st.button("보호된 이미지 다운로드하기")
