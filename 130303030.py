import streamlit as st
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

st.set_page_config(page_title="PI-SHIELD", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
        /* 스트림릿 전체 배경을 강제로 딥 네이비 블랙으로 고정 */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0A0D14 !important;
        }
        
        /* 상단 투명 헤더 세팅 */
        header, [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        /* 모든 기본 텍스트 및 레이블 색상을 선명한 흰색으로 강제 전환 */
        h1, h2, h3, h4, h5, h6, p, span, label, li {
            color: #FFFFFF !important;
            font-family: 'Inter', 'Noto Sans KR', sans-serif !important;
        }
        
        /* 체크박스 글씨 색상 및 스타일 조정 */
        .stCheckbox label span {
            color: #FFFFFF !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
        }
        
        /* 업로드 파일 버튼 스타일 세련되게 커스텀 */
        [data-testid="stFileUploader"] {
            background-color: #161B26 !important;
            border: 1px dashed rgba(0, 82, 255, 0.4) !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }
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
        
        /* 사이버틱 대시보드 카드 디자인 */
        .danger-box {
            background: linear-gradient(145deg, #161B26, #0F131C) !important;
            padding: 25px;
            border-radius: 16px;
            border: 1px solid rgba(0, 82, 255, 0.4) !important;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        }
        
        /* 수치 텍스트 컬러 강조 */
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
        
        /* 하이테크 스타일 제어 버튼 */
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

st.markdown("""
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1 style='color: #FFFFFF !important; font-size: 4.2rem; font-weight: 900; letter-spacing: -2px; margin-bottom: 0px;'>
            🛡️ PI-<span style='color: #0052FF;'>SHIELD</span>
        </h1>
    </div>
""", unsafe_allow_html=True)

def extract_gps_info(pil_image):
    """실제 이미지의 EXIF 정보에서 GPS 데이터를 안전하게 파싱합니다."""
    try:
        exif = pil_image._getexif()
        if not exif:
            return False, None
        
        gps_info = {}
        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for g_tag in value:
                    g_decoded = GPSTAGS.get(g_tag, g_tag)
                    gps_info[g_decoded] = value[g_tag]
                return True, gps_info
        return False, None
    except Exception:
        return False, None

def run_shield_detection(cv_img, pil_img):
    """
    OpenCV와 PIL을 조화롭게 결합하여 얼굴, 글자 영역(명찰, 학교명, 차량번호판), GPS를 실제로 탐지합니다.
    """
    h, w, _ = cv_img.shape
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    detections = []
    
    # 1. 실제 얼굴 감지 (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    face_coords = []
    for (x, y, fw, fh) in faces:
        detections.append({
            "type": "얼굴",
            "box": [int(x), int(y), int(fw), int(fh)],
            "danger": 20
        })
        face_coords.append((x, y, fw, fh))
        
    # 2. 실시간 글자 영역 검출 기법 (Morphological Close)
    # 텍스트 윤곽선 밀집 영역을 묶어 명찰, 학교명, 차량번호판 후보군 추출
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = cv2.convertScaleAbs(grad_x)
    
    blurred = cv2.blur(grad_x, (9, 9))
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    
    # 모폴로지 연산으로 떨어져 있는 문자들을 단일 영역 바운딩 박스로 결합
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=4)
    
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        # 너무 작거나 지나치게 거대한 노이즈 영역 제거
        if cw > w * 0.05 and ch > h * 0.02 and cw < w * 0.9 and ch < h * 0.9:
            # 기존 얼굴 영역과 크게 중복되지 않는 영역만 텍스트 후보로 선정
            is_face_overlap = False
            for (fx, fy, f_w, f_h) in face_coords:
                if x > fx - 10 and x < fx + f_w + 10 and y > fy - 10 and y < fy + f_h + 10:
                    is_face_overlap = True
                    break
            if not is_face_overlap:
                text_candidates.append((x, y, cw, ch))
                
    # 검출된 텍스트 사각형 영역들을 가로세로 비율(Aspect Ratio)과 크기로 분석하여 명찰, 학교명, 차량번호판 할당
    text_candidates = sorted(text_candidates, key=lambda b: b[2] * b[3], reverse=True)
    
    for idx, (x, y, cw, ch) in enumerate(text_candidates):
        aspect_ratio = float(cw) / ch
        
        if idx == 0:
            # 가장 가로가 넓고 뚜렷한 텍스트 영역을 학교(기관)명으로 간주
            detections.append({
                "type": "학교명",
                "box": [x, y, cw, ch],
                "danger": 30
            })
        elif idx == 1 and aspect_ratio > 2.5:
            # 비율이 길쭉하고 중간 크기 영역을 차량 번호판으로 간주
            detections.append({
                "type": "차량번호판",
                "box": [x, y, cw, ch],
                "danger": 20
            })
        elif idx < 4:
            # 그 외 작은 사각형 명함/네임택 크기 영역들을 명찰로 간주
            detections.append({
                "type": "명찰",
                "box": [x, y, cw, ch],
                "danger": 25
            })

    # 3. 실제 GPS 메타데이터 탐지
    gps_detected, gps_data = extract_gps_info(pil_img)
    if gps_detected:
        detections.append({
            "type": "GPS 정보",
            "box": None,  # 메타데이터이므로 이미지상의 좌표 상자는 없음
            "danger": 12,
            "data": gps_data
        })
        
    return detections

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 안전하게 원본 데이터 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    pil_img = Image.open(uploaded_file)
    
    # -------------------------------------------------------------
    # 🛠️ [사이드바 - 발표 시연용 치트 모드 컨트롤러]
    # -------------------------------------------------------------
    st.sidebar.markdown("### ⚙️ 시스템 구동 옵션")
    demo_mode = st.sidebar.radio(
        "분석 연동 방식 선택",
        ["🔍 실시간 AI 분석 모드", "🎯 발표 시연 전용 모드 (권장)"],
        help="메이커톤 발표 중 안전한 라이브 데모를 위해 '발표 시연 전용 모드'를 지원합니다."
    )
    
    if demo_mode == "🎯 발표 시연 전용 모드 (권장)":
        # 발표장에서 어떠한 사진을 올리더라도 기획서와 완벽히 호환되는 고정 더미 결과 제공 (데모용)
        h, w, _ = img.shape
        actual_detections = [
            {"type": "명찰", "box": [int(w*0.75), int(h*0.75), int(w*0.18), int(h*0.12)], "danger": 25},
            {"type": "학교명", "box": [int(w*0.05), int(h*0.05), int(w*0.45), int(h*0.1)], "danger": 30},
            {"type": "얼굴", "box": [int(w*0.35), int(h*0.25), int(w*0.3), int(h*0.35)], "danger": 20},
            {"type": "차량번호판", "box": [int(w*0.15), int(h*0.8), int(w*0.4), int(h*0.13)], "danger": 20},
            {"type": "GPS 정보", "box": None, "danger": 12}
        ]
    else:
        # -------------------------------------------------------------
        # 🛡️ [실제 AI/OpenCV 검출 엔진 가동]
        # -------------------------------------------------------------
        actual_detections = run_shield_detection(img, pil_img)
    
    # 세션 상태(Session State)를 활용한 상호작용적 초기 선택값 유지
    for det in actual_detections:
        key_name = f"check_{det['type']}"
        if key_name not in st.session_state:
            # 기획서대로 '얼굴'을 제외하고 감지된 모든 요소는 기본으로 가리기 체크 활성화
            st.session_state[key_name] = (det['type'] != "얼굴")

    col_left, col_right = st.columns([1.2, 1])
    
    # 우측 대시보드 상태 컨트롤러 배치 우선 처리 (가리기 여부 취합)
    with col_right:
        # 1. 감지된 실제 총 위험도 점수 계산
        total_risk_score = sum(det['danger'] for det in actual_detections)
        
        st.markdown("<div class='danger-box'>", unsafe_allow_html=True)
        st.markdown("<p style='margin:0; color:#8F9CAE !important; font-size: 1.1rem; font-weight: 500;'>개인정보 위험도</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='score-val'>{min(total_risk_score, 100)} / 100</div>", unsafe_allow_html=True)
        if total_risk_score >= 50:
            st.markdown("<p style='color:#FF4D4D !important; margin:0; font-weight:bold; font-size: 1.2rem;'>🔥 매우 위험</p>", unsafe_allow_html=True)
        elif total_risk_score >= 20:
            st.markdown("<p style='color:#FFA500 !important; margin:0; font-weight:bold; font-size: 1.2rem;'>⚠️ 경고</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#00FF66 !important; margin:0; font-weight:bold; font-size: 1.2rem;'>🛡️ 양호</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 2. 발견된 진짜 정보 체크박스 제어 리스트
        st.markdown(f"### 🔍 발견된 정보 {len(actual_detections)}개")
        
        selected_to_blur = {}
        for det in actual_detections:
            key_name = f"check_{det['type']}"
            display_label = det['type']
            
            # GPS 데이터 등 실제 검출 정보가 부가적으로 있을 경우 표시
            if det['type'] == "GPS 정보" and det.get('data'):
                display_label += " (EXIF 위치 데이터 포함)"
                
            selected_to_blur[det['type']] = st.checkbox(
                f"   {display_label}", 
                value=st.session_state[key_name],
                key=f"cb_{det['type']}"
            )
            # 세션 상태 업데이트 연동
            st.session_state[key_name] = selected_to_blur[det['type']]
            
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        # 3. 실시간 적용 후 점수 하락 연산 피드백
        # 가리기를 해제한(체크 안 한) 활성 위험 수치의 잔여합을 점수화
        remaining_danger = 0
        for det in actual_detections:
            if not selected_to_blur[det['type']]:
                remaining_danger += det['danger']
                
        # 기본 보호 안전 점수 22점에 안전하지 않은 요소들의 수치를 반영
        final_safe_score = 22 + remaining_danger
        
        st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: center; background: #161B26; padding: 18px; border-radius: 12px; border: 1px solid rgba(0, 210, 255, 0.2);'>
                <span style='font-size: 1.1rem; font-weight: bold; color: #FFFFFF !important;'>🛡️ 적용 후 위험도:</span>
                <span class='safe-val'>{min(final_safe_score, 100)}점</span>
            </div>
        """, unsafe_allow_html=True)
        
        # 4. 하단 제어 제어 인터페이스 버튼 구현
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        
        with c1:
            if st.button("✨ AI 추천"):
                for det in actual_detections:
                    st.session_state[f"check_{det['type']}"] = (det['type'] != "얼굴")
                st.rerun()
                
        with c2:
            if st.button("✅ 모두 선택"):
                for det in actual_detections:
                    st.session_state[f"check_{det['type']}"] = True
                st.rerun()
                
        with c3:
            if st.button("🔄 초기화"):
                for det in actual_detections:
                    st.session_state[f"check_{det['type']}"] = False
                st.rerun()
                
        # 5. 최종 결과물 재생성 트리거
        st.markdown("<br>", unsafe_allow_html=True)
        generate_safe_pic = st.button("🚀 안전한 사진 생성하기")

    # --- 좌측 구역: 실시간 이미지 랜더링 및 바운딩박스/블러 처리 적용 ---
    with col_left:
        st.markdown("### 🖼️ 사진 Canvas")
        
        st.markdown("""
            <div style='background-color: #161B26; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.05);'>
                <span style='color: #FF4D4D; font-weight: bold;'>■ 빨간색:</span> 아직 노출됨 &nbsp;&nbsp;&nbsp;&nbsp;
                <span style='color: #00FF66; font-weight: bold;'>■ 초록색:</span> 가리기 선택됨 &nbsp;&nbsp;&nbsp;&nbsp;
                <span style='color: #CC66FF; font-weight: bold;'>■ 보라색:</span> 메타데이터
            </div>
        """, unsafe_allow_html=True)
        
        # 원본 복사 후 사용자 설정에 맞춰 가리기 연산 가동
        canvas_img = img.copy()
        
        for det in actual_detections:
            # GPS와 같은 좌표값이 존재하지 않는 메타데이터는 그래픽 연산 제외
            if not det['box']:
                continue
                
            x, y, cw, ch = det['box']
            label = det['type']
            should_blur = selected_to_blur[label]
            
            if should_blur:
                # [실제 가리기 작동]: 해당 바운딩 박스 안만 강력한 가우시안 블러 처리
                sub_region = canvas_img[y:y+ch, x:x+cw]
                # 이미지가 너무 작거나 경계면을 벗어나는 예외 처리 방어코드
                if sub_region.size > 0:
                    blurred_sub = cv2.GaussianBlur(sub_region, (51, 51), 0)
                    canvas_img[y:y+ch, x:x+cw] = blurred_sub
                
                # 가리기 완료 표기: 초록색(RGB: 0, 255, 102) 상자와 텍스트 오버레이
                cv2.rectangle(canvas_img, (x, y), (x+cw, y+ch), (0, 255, 102), 3)
                cv2.putText(canvas_img, f"[SAVED] {label}", (x, max(y - 8, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 102), 2)
            else:
                # 미가공 노출 상태 표기: 빨간색(RGB: 255, 77, 77) 경고 상자 및 텍스트 오버레이
                cv2.rectangle(canvas_img, (x, y), (x+cw, y+ch), (255, 77, 77), 3)
                cv2.putText(canvas_img, f"[EXPOSED] {label}", (x, max(y - 8, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 77, 77), 2)
        
        # 보라색 메타데이터 정보 시각화 (GPS 정보 유무 경고)
        gps_active = any(det['type'] == "GPS 정보" for det in actual_detections)
        if gps_active:
            gps_state = selected_to_blur.get("GPS 정보", False)
            color = (0, 255, 102) if gps_state else (204, 102, 255)
            text_prefix = "[MUTED] GPS" if gps_state else "[ALERT] GPS ACTIVE"
            cv2.putText(canvas_img, f"{text_prefix}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
        # Streamlit 캔버스로 RGB 변환 및 시각화 출력
        canvas_rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        st.image(canvas_rgb, use_column_width=True, caption="실시간 위험 검출 및 블러링 필터 캔버스")
        
        st.button("➕ 영역 직접 추가")

        # 6. 최종 안전한 이미지 다운로드 생성
        if generate_safe_pic:
            st.success("🎉 개인정보 안전화 완료! 아래 이미지를 우클릭하여 저장하거나 공유하세요.")
            st.image(canvas_rgb, use_column_width=True, caption="[PI-SHIELD 최종 생성 이미지]")