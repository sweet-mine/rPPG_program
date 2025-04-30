'''
last modified date = 2024-11-14
version = 0.2
use = 검출된 RGB값을 시각화하기 위해 사용하는 클래스, 함수 정의
TODO : scipy로 밴드패스필터
'''

import cv2
import numpy
import matplotlib.pyplot as plt
from collections import deque
import time
import constantValue

class RealTimeRGBPlot:
    def __init__(self, max_len=constantValue.dequeMaxLen, xValue=constantValue.xValue):
        self.xValue = xValue
        self.max_len = max_len
        self.array_RGB_prev = None
        self.r_data = deque(maxlen=max_len)
        self.g_data = deque(maxlen=max_len)
        self.b_data = deque(maxlen=max_len)
        self.total_data = deque(maxlen=max_len)
        self.time = 0
        self.time_data = deque(maxlen=max_len)

        # Plot 설정
        self.fig, self.ax = plt.subplots()
        self.r_line, = self.ax.plot([], [], 'r-', label='Red')
        self.g_line, = self.ax.plot([], [], 'g-', label='Green')
        self.b_line, = self.ax.plot([], [], 'b-', label='Blue')
        self.total_line, = self.ax.plot([], [], 'p-', label='total')  # test
        self.ax.set_xlim(0, xValue)
        self.ax.set_ylim(0, 30)
        self.ax.set_title("Real-time RGB Signal")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Intensity")
        self.ax.legend()
        plt.ion()

    def append_rgb_data(self, array_RGB):
        """RGB 값을 기반으로 deque에 차이값 저장"""
        if self.array_RGB_prev is None:
            self.array_RGB_prev = array_RGB
            return  # 초기 상태에서는 변화값이 없음

        r_diff = abs(self.array_RGB_prev[0] - array_RGB[0])
        g_diff = abs(self.array_RGB_prev[1] - array_RGB[1])
        b_diff = abs(self.array_RGB_prev[2] - array_RGB[2])
        total_diff = r_diff + (2 * g_diff) + (0.5 * b_diff)

        self.r_data.append(r_diff)
        self.g_data.append(g_diff)
        self.b_data.append(b_diff)
        self.total_data.append(total_diff)

        self.time = time.perf_counter()
        self.time_data.append(self.time)
        self.array_RGB_prev = array_RGB

    def update_plot(self):
        """현재 deque 데이터를 사용하여 그래프를 업데이트"""
        self.r_line.set_data(self.time_data, self.r_data)
        self.g_line.set_data(self.time_data, self.g_data)
        self.b_line.set_data(self.time_data, self.b_data)
        self.total_line.set_data(self.time_data, self.total_data)

        if self.time > self.xValue:
            self.ax.set_xlim(self.time - self.xValue, self.time)

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