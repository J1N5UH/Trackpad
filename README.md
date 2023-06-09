## Camera와 Mediapipe를 이용하여 TrackPad를 구현하였습니다.

![GIF](trackpad.gif)

### 총 5단계를 거쳐서 진행한 Project이며, 각 단계는 아래와 같습니다.

#### 1. 학습 데이터 수집
#### 2. 학습 데이터 전처리 과정
#### 3. 학습 데이터 학습
#### 4. 학습데이터 검증
#### 5. TrackPad 구현

# Mediapipe 소개
Mediapipe는  Google에서 개발한 오픈 소스 프레임워크로, 컴퓨터 비전 및 머신 러닝 기술을 쉽게 구현하고 배포할 수 있도록 도와주는 도구입니다. MediaPipe를 사용하면 비디오, 오디오, 이미지 등 다양한 형식의 데이터를 처리하고 분석하는 데 사용할 수 있습니다.
아래 사진과 같이 각 관절마다 번호를 부여하여 각 관절의 위치를 좌표값으로 받아와 마우스의 커서와 기능을 구현하였습니다.
![PNG](Mediapipe.png)

# 1. 학습 데이터 수집

학습 데이터는 직접 카메라로 다양한 제스처를 찍으며 총 1024개의 영상을 직접 만들어서 제작 하였습니다.
영상을 제작 할 때는 한 가지의 제스처만 촬영을 하며 딱 한번의 행동만들 촬영하였습니다.

# 2. 학습 데이터 전처리 과정

1번 과정에서 생성한 데이터들을 Numpy 형태의 데이터로 추출을 합니다.

Numpy에는 index 0번 부터 index 20번 까지의 x, y좌표값을 받아오며 프레임 단위로 Numpy 배열을 생성합니다.

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 각 랜드마크의 x, y 좌표 추출
            landmarks = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.extend([cx, cy])
            
            return np.array(landmarks)
            
여기에서 추출되는 데이터는 다음과 같습니다.

1. 손가락 각도 데이터: 15개
2. lanmark 데이터: 63개
3. 랜드마크 visivility: 21개
4. 정답 라벨: 1개

# 3. 학습 데이터 학습

2번에서 생성한 데이터를 Tensorflow 학습모델을 이용하여 학습을 합니다.
각 제스처마다 따로 학습을 시켜서 라벨링을 하고 이후 하나의 모델로 합치는 과정입니다.


    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLRonPlateau

        history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=200,
        callbacks=[
            ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
        ]
        )
       
# 4. TrackPad 구현

최종적으로 구현하는 단계 입니다.

학습된 모델을 불러와서 카메라에서 탐지한 제스처의 행위를 인식하여 기능을 수행합니다.


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


