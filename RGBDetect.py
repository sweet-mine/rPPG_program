'''
last modified date = 2024-11-14
version = 0.2
use = RGB값을 검출하고 시각화하기 위해 사용하는 클래스, 함수 정의
TODO : scipy로 밴드패스필터
'''

import cv2
import numpy
import matplotlib.pyplot as plt
from collections import deque
import time
import constantValue

class RealTimeRGBPlot: # RGB array를 입력 받아 bandpass filtering을 한 후 시각화하기 위한 클래스
    def __init__(self, max_len = constantValue.dequeMaxLen, xValue = constantValue.xValue):
        # 데이터 저장을 위한 deque 초기화
        self.xValue = xValue
        self.max_len = max_len
        self.array_RGB_prev = None
        self.r_data = deque(maxlen=max_len)
        self.g_data = deque(maxlen=max_len)
        self.b_data = deque(maxlen=max_len)
        self.total_data = deque(maxlen=max_len) #test
        self.time = 0
        self.time_data = deque(maxlen=max_len)

        # 플롯 설정
        self.fig, self.ax = plt.subplots()
        self.r_line, = self.ax.plot([], [], 'r-', label='Red')
        self.g_line, = self.ax.plot([], [], 'g-', label='Green')
        self.b_line, = self.ax.plot([], [], 'b-', label='Blue')
        self.total_line, = self.ax.plot([], [], 'p-', label='total') #test
        self.ax.set_xlim(0, xValue)
        self.ax.set_ylim(0, 30)  # RGB 값 범위 (-20~20)
        self.ax.set_title("Real-time RGB Signal")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Intensity")
        self.ax.legend()
        plt.ion()  # 인터랙티브 모드 활성화

    def update_data(self, array_RGB):
        if self.array_RGB_prev == None: #초기값
            self.array_RGB_prev = array_RGB
            return

        # 이전 RGB값과의 차이를 계산하여 업데이트, 시간축 업데이트
        self.r_data.append(abs(self.array_RGB_prev[0] - array_RGB[0]))
        self.g_data.append(abs(self.array_RGB_prev[1] - array_RGB[1]))
        self.b_data.append(abs(self.array_RGB_prev[2] - array_RGB[2]))
        self.total_data.append(self.r_data[-1] + (2 * self.g_data[-1]) + (0.5 * self.b_data[-1])) #test
        self.array_RGB_prev = array_RGB
        self.time = time.perf_counter()
        self.time_data.append(self.time) # 현실 시간으로 x축 반영

        # 각 시각화 라인 데이터 업데이트
        self.r_line.set_data(self.time_data, self.r_data)
        self.g_line.set_data(self.time_data, self.g_data)
        self.b_line.set_data(self.time_data, self.b_data)
        self.total_line.set_data(self.time_data, self.total_data) #test

        # x축 업데이트 (시간 경과에 따라 이동)
        if self.time > self.xValue:
            self.ax.set_xlim(self.time - self.xValue, self.time)

        # 그래프 업데이트
        plt.pause(0.05)
        plt.draw()

    def close(self):
        plt.ioff()
        plt.show()

def returnRGB(frame_rightcheek, frame_leftcheek): # 양쪽 뺨 rgb값 더한 후 2로 나누고 리턴하는 함수
    bgr_right = numpy.array(cv2.mean(frame_rightcheek))
    bgr_left = numpy.array(cv2.mean(frame_leftcheek))
    bgr_all = (bgr_left + bgr_right) / 2
    mean_rgb = (bgr_all[2], bgr_all[1], bgr_all[0])
    return mean_rgb