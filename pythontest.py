import sys
import cv2
import numpy as np
import RGBDetect
import constantValue
import time
import serial
import mediapipe as mp

# 시리얼통신 객체 생성
heratRateSerial = serial.Serial(port=constantValue.serialPort, baudrate=115200)

cnt = constantValue.cnt
cap = cv2.VideoCapture(0)

# RGB값 저장용 클래스 생성
plot = RGBDetect.RealTimeRGBPlot()

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

heartRate = None

# 프레임 및 시리얼 초기화
ret, frame = cap.read()
cv2.imshow("Result", frame)
heratRateSerial.reset_input_buffer()

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

    # 심박수 수신
    if heratRateSerial.in_waiting > 0:
        heartRate = heratRateSerial.readline().decode().strip()
        print(heartRate)

    if (heartRate == None):
        continue

    if heartRate == "Wait for valid data !":
        print("\n심박수를 놓쳐 종료합니다.\n")
        sys.exit(1)

    plot.append_rgb_data(RGBarray)
    plot.append_heart_data(int(heartRate))

    cnt += 1

    if plot.isFull():
        break

    try:
        cv2.imshow("Result", frame)
    except:
        continue

    if cv2.waitKey(1) > 0:
        sys.exit(0)

print("\n모든 데이터를 수집했습니다.\n")
plot.save_plot()
plot.save_dataset()
