'''
last modified date = 2024-11-14
version = 0.1
use = main
'''


import cv2
import numpy
import RGBDetect
import constantValue
import time

cnt = constantValue.cnt # 얼굴인식 프레임
cap = cv2.VideoCapture(0)

# 얼굴 검출 분류기 로드
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#RGB값 시각화용 plot 생성
plot = RGBDetect.RealTimeRGBPlot()

# 비디오 재생 시작
while True:
    ret, frame = cap.read()     # 카메라로부터 현재 영상을 받아 frame에 저장, 잘 받았다면 ret가 참

    if cnt >= constantValue.cnt: # value.cnt번째 프레임마다 얼굴 영역 다시 계산
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor= 1.1, minNeighbors=4, minSize=(20,20))
        if type(faces) == numpy.ndarray:
            cnt = 0

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

        plot.update_data(RGBarray)
        cnt += 1
        # what?

    # 결과 비디오 출력
    try:
        cv2.imshow("Result", frame)
    except Exception as e:
        continue

    #아무 키나 입력되면 멈추기
    if cv2.waitKey(1) > 0:
        break