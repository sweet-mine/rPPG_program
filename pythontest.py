'''
last modified date = 2025-05-10
version = 0.2
use = main
'''

import sys
import cv2
import numpy
import RGBDetect
import constantValue
import time
import serial

#시리얼통신 객체 생성
heratRateSerial = serial.Serial(port = constantValue.serialPort, baudrate=115200)

cnt = constantValue.cnt # 얼굴인식 프레임
cap = cv2.VideoCapture(0)

# 얼굴 검출 분류기 로드
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#RGB값 저장용 클래스 생성
plot = RGBDetect.RealTimeRGBPlot()

#웹캠 최대 프레임 체크
print(cap.get(cv2.CAP_PROP_FPS))

#심박 정보(String)
heartRate = None;

# 비디오 재생 시작
while True:
    ret, frame = cap.read()     # 카메라로부터 현재 영상을 받아 frame에 저장, 잘 받았다면 ret가 참

    if cnt >= constantValue.cnt: # value.cnt번째 프레임마다 얼굴 영역 다시 계산, 실패하면 종료
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor= 1.1, minNeighbors=4, minSize=(20,20))
        if type(faces) == numpy.ndarray:
            cnt = 0
        else:
            print("\n얼굴 인식을 놓쳐 종료합니다.\n")
            sys.exit(1)

    if type(faces) == numpy.ndarray:  # faces 검출시 실행
        x, y, w, h = max(faces, key=lambda x: x[2] * x[3])  # 가장 큰 얼굴 객체만 검출 후 좌표 얻어냄
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
        frame_face = frame[y:y + h, x:x + w].copy() # 검출된 얼굴 부분 자르기

        #화면 기준 왼쪽 뺨 영역 생성
        lefty = int(y + (h / 2))
        leftyh = int(lefty + (h / 8))
        leftx = int(x + (w * 0.2))
        leftxw = int(leftx + (w / 8))
        #cv2.rectangle(frame, (leftx, lefty), (leftxw, leftyh), (255, 255, 255), thickness=2)
        frame_leftCheek = frame[lefty: leftyh, leftx: leftxw].copy()

        #화면 기준 오른쪽 뺨 영역 생성
        righty = int(y + (h / 2))
        rightyh = int(righty + (h / 8))
        rightx = int(x + (w * 0.7))
        rightxw = int(rightx + (w / 8))
        #cv2.rectangle(frame, (rightx, righty), (rightxw, rightyh), (255, 255, 255), thickness=2)
        frame_rightCheek = frame[righty: rightyh, rightx: rightxw].copy()

        RGBarray = RGBDetect.returnRGB(frame_rightCheek, frame_leftCheek) # 양쪽 뺨 RGB값 계산

        # 심박수 체크
        if heratRateSerial.in_waiting > 0:
            heartRate = heratRateSerial.readline().decode().strip()
            print(heartRate)
        # 아직 못 받아왔을 경우 다시
        if(heartRate == None):
            continue
        # 심박수 놓치면 종료
        if(heartRate == "Wait for valid data !"):
            print("\n심박수를 놓쳐 종료합니다.\n")
            sys.exit(1)

        plot.append_rgb_data(RGBarray)
        plot.append_heart_data(int(heartRate))

    cnt += 1

    #지정된 값만큼 데이터 쌓였으면 break
    if plot.isFull():
        break

    # 비디오 출력
    try:
        cv2.imshow("Result", frame)
    except Exception as e:
        continue

    #아무 키나 입력되면 멈추기
    if cv2.waitKey(1) > 0:
        sys.exit(0)

print("\n모든 데이터를 수집했습니다.\n")
plot.save_dataset() #데이터셋으로 저장 후 종료
