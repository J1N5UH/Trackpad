import cv2
import mediapipe as mp
import numpy as np
import os

# Mediapipe hands 모델 생성
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def process_frame(frame):
    # RGB로 이미지 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 손의 랜드마크(landmark) 처리
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 각 랜드마크의 x, y 좌표 추출
            landmarks = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.extend([cx, cy])
            
            return np.array(landmarks)

    return None

# 학습 데이터가 저장된 경로
videos_path = '여기 경로'

# 액션의 라벨
labels = ['single_click', 'double_click', 'scroll_up', 'scroll_down']

# 처리된 데이터와 라벨을 저장할 배열 초기화
data, target = [], []

for label in labels:
    print(f'{label} 비디오 처리중...')

    for file_name in os.listdir(os.path.join(videos_path, label)):
        video_path = os.path.join(videos_path, label, file_name)
        
        # 각 비디오 파일로부터 프레임을 읽어옴
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임에서 랜드마크 추출
            landmarks = process_frame(frame)
            if landmarks is not None:
                data.append(landmarks)
                target.append(label)

        cap.release()

data = np.array(data)
target = np.array(target)

# numpy 파일로 데이터와 라벨 저장
np.save('data.npy', data)
np.save('target.npy', target)
