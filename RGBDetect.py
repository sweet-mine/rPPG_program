'''
last modified date = 2025-05-10
version = 0.3
use = ROI를 제공하면 검출된 RGB값과 심박수값을 저장하고 시각화 할 수 있는 클래스, 함수 정의
'''

import os
import pandas as pd
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
        self.heart_data = deque(maxlen=max_len)

        # Plot 설정
        self.fig, self.ax = plt.subplots()
        self.r_line, = self.ax.plot([], [], 'r-', label='Red')
        self.g_line, = self.ax.plot([], [], 'g-', label='Green')
        self.b_line, = self.ax.plot([], [], 'b-', label='Blue')
        self.total_line, = self.ax.plot([], [], 'p-', label='total')  # test
        self.ax.set_xlim(0, xValue)
        self.ax.set_ylim(0, 10)
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

    def append_heart_data(self, heartRate):
        self.heart_data.append(heartRate)

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

    def save_dataset(self):
        # 저장 폴더와 기본 파일 이름
        folder = "Dataset"
        base_name = "data"

        # 폴더 없으면 생성
        os.makedirs(folder, exist_ok=True)

        # 현재 폴더 안에 있는 파일 숫자 파악
        existing_files = os.listdir(folder)
        existing_numbers = [
            int(f.split('_')[1].split('.')[0])
            for f in existing_files
            if f.startswith(base_name) and f.endswith('.csv') and f.split('_')[1].split('.')[0].isdigit()
        ]

        next_number = max(existing_numbers) + 1 if existing_numbers else 1
        filename = f"{base_name}_{next_number}.csv"
        filepath = os.path.join(folder, filename)

        # 저장
        df = pd.DataFrame({
            'feature': list(self.total_data),
            'label': list(self.heart_data)
        })
        df.to_csv(filepath, index=False)

        print(f"Saved to: {filepath}")

    def isFull(self):
        if len(self.total_data) == self.total_data.maxlen:
            return True
        else:
            return False

    def close(self):
        plt.ioff()
        plt.show()


def returnRGB(frame_rightcheek, frame_leftcheek): # 양쪽 뺨 rgb값 더한 후 2로 나누고 리턴하는 함수
    bgr_right = numpy.array(cv2.mean(frame_rightcheek))
    bgr_left = numpy.array(cv2.mean(frame_leftcheek))
    bgr_all = (bgr_left + bgr_right) / 2
    mean_rgb = (bgr_all[2], bgr_all[1], bgr_all[0])
    return mean_rgb