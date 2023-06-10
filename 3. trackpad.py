import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui

# Mediapipe hands 모델 생성
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 훈련된 모델 불러오기
model = load_model('hand_gesture_model')

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

cap = cv2.VideoCapture(0)  # 웹캠 사용

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = process_frame(frame)
    if landmarks is not None:
        landmarks = np.expand_dims(landmarks, axis=0)  # 모델에 입력하기 위해 차원을 맞춰줍니다.
        action = model.predict(landmarks)
        action = np.argmax(action)  # 가장 높은 확률을 가진 클래스의 인덱스를 얻습니다.

        # 동작 실행
        if action == 0:
            pyautogui.click(clicks=1)  # 한 번 클릭
        elif action == 1:
            pyautogui.click(clicks=2)  # 두 번 클릭
        elif action == 2:
            pyautogui.scroll(1)  # 스크롤 업
        elif action == 3:
            pyautogui.scroll(-1)  # 스크롤 다운

cap.release()
cv2.destroyAllWindows()
