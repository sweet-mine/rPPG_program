import sys
import cv2
import numpy as np
import RGBDetect
import constantValue
import mediapipe as mp
import torch
import cnnGru as cnn


######## 추론용 파라미터 ########
# 로컬 파일명 (현재 코드와 같은 폴더에 저장됨)
model_path = 'model_result.pth'
scaler_path = 'feature_scaler.pkl'

# 학습 시 사용했던 파라미터터 값들
hr_min, hr_max = 40, 200  # 최소/최대 심박수 (역정규화 시 사용)
sequence_length = 50  # 시퀀스 길이
input_channels = 1  # 모델 학습 시 사용된 입력 채널 수 (rgb로부터 계산된 값 하나가 입력이므로 1)

# 디바이스 설정 (CUDA 가능하면 GPU 사용, 아니면 CPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




######## 객체 생성 및 초기화 ########
# 모델 및 스케일러 생성
model, scaler = cnn.load_model_and_scaler(model_path, scaler_path, input_channels, device)

cnt = constantValue.cnt
cap = cv2.VideoCapture(0)

# RGB값 저장용 객체 생성
plot = RGBDetect.RealTimeRGBPlot()

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 프레임 초기화
ret, frame = cap.read()
cv2.imshow("Result", frame)

frame_h, frame_w = frame.shape[:2]

# 좌표 평균 함수
def get_average_landmark_coords(landmarks, indices, width, height):
    coords = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in indices])
    return np.mean(coords, axis=0).astype(int)

# 비디오 루프
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        print("\n얼굴 인식을 놓쳐 종료합니다.\n")
        sys.exit(1)

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape

    # 뺨 영역 평균 위치 계산
    left_cheek_ids = [50, 205]
    right_cheek_ids = [280, 425]
    left_cheek = get_average_landmark_coords(landmarks, left_cheek_ids, w, h)
    right_cheek = get_average_landmark_coords(landmarks, right_cheek_ids, w, h)

    # 뺨 영역 박스 크기 (20x20 사각형)
    box_size = 20
    lx, ly = left_cheek
    rx, ry = right_cheek

    left_box = frame[ly:ly + box_size, lx:lx + box_size].copy()
    right_box = frame[ry:ry + box_size, rx:rx + box_size].copy()

    # 디버깅용 사각형 표시
    cv2.rectangle(frame, (lx, ly), (lx + box_size, ly + box_size), (0, 255, 0), 2)
    cv2.rectangle(frame, (rx, ry), (rx + box_size, ry + box_size), (0, 255, 0), 2)

    # RGB 추출
    RGBarray = RGBDetect.returnRGB(right_box, left_box)

    plot.append_rgb_data(RGBarray)

    cnt += 1

    # 필요한 만큼 데이터 수집했으면 예상 심박수 출력
    if plot.isFull():
        bpm = cnn.predict_bpm_from_sequence(
            np.array(plot.total_data),  # rPPG 시계열 데이터 (NumPy 배열)
            model,  # 애플리케이션 시작 시 로드된 PyTorch 모델 객체
            scaler,  # 애플리케이션 시작 시 로드된 스케일러 객체
            device,  # 설정된 연산 장치 (cpu 또는 cuda)
            sequence_length,  # 모델이 기대하는 입력 시퀀스 길이
            hr_min,  # 역정규화에 사용될 최소 심박수
            hr_max  # 역정규화에 사용될 최대 심박수
        )
        print(bpm)
        plot.total_data.clear()

    try:
        cv2.imshow("Result", frame)
    except:
        continue

    if cv2.waitKey(1) > 0:
        sys.exit(0)