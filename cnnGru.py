import torch
import torch.nn as nn
import numpy as np
import joblib
import sklearn

# 1. CNN-GRU 모델 정의 (모델 학습 때 사용했던 것과 동일한 구조)
class CNNGRUModel(nn.Module):
    def __init__(self, input_channels=1):
        super(CNNGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x 형태: (배치 크기, 채널 수, 시퀀스 길이)
        x = self.conv1(x)
        x = self.relu(x)
        # conv1d 이후 x 형태: (배치 크기, 출력 채널 수, 시퀀스 길이)
        # GRU는 다음 형태를 기대: (배치 크기, 시퀀스 길이, 입력 크기)
        x = x.permute(0, 2, 1)  # 차원 순서 변경
        x, _ = self.gru(x)
        # gru 이후 x 형태: (배치 크기, 시퀀스 길이, 은닉층 크기)
        # 마지막 타임 스텝의 출력을 사용
        return self.fc(x[:, -1, :])


# 2. 모델 및 스케일러 로드 함수
def load_model_and_scaler(model_path, scaler_path, input_channels, device):
    """학습된 모델과 스케일러를 로드"""
    try:
        model = CNNGRUModel(input_channels=input_channels).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))  # map_location으로 어떤 디바이스에서도 로드 가능하게 설정
        model.eval()  # 추론 모드로 설정
        print(f"모델 로드 완료: {model_path}")

        feature_scaler = joblib.load(scaler_path)
        print(f"스케일러 로드 완료: {scaler_path}")
        return model, feature_scaler
    except FileNotFoundError as e:
        print(f"오류: 로컬 파일({e.filename})을 찾을 수 없습니다.")
        return None, None
    except Exception as e:
        print(f"모델 또는 스케일러 로드 중 오류 발생: {e}")
        return None, None


# -------------------- 실제로 심박수 예측하는 코드 --------------------

# 3. 심박수 예측 함수
# 학습한 모델로부터 나온 정보 이용함 : model 매개변수는 위의 load_model_and_scaler 함수로부터 나온 객체
def predict_bpm_from_sequence(raw_sequence_data, model, scaler, device, sequence_length, hr_min, hr_max):
    if len(raw_sequence_data) != sequence_length:
        print(f"오류: 입력 데이터 길이가 {len(raw_sequence_data)}이지만, 모델은 {sequence_length}를 기대합니다.")
        return None
    sequence_data_reshaped = np.array(raw_sequence_data).reshape(-1, 1)
    try:
        scaled_data = scaler.transform(sequence_data_reshaped)
    except Exception as e:
        print(f"데이터 스케일링 중 오류 발생: {e}")
        return None
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).T
    input_tensor = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor).squeeze()
        scaled_prediction = output.cpu().item()
    prediction = scaled_prediction * (hr_max - hr_min) + hr_min
    return prediction

"""
추가 메모)
모델로부터 예측된 심박수와 실제 심박수를 비교하려면 갤럭시 워치와 같은 심박수 측정 장치 필요
모델과 측정 장치 동시에 돌리면서 결과 비교하면서 봐야 함 
"""