from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 저장된 numpy 데이터 로드
data = np.load('data.npy')
target = np.load('target.npy')

# 라벨 인코딩 - 문자열 라벨을 숫자로 변환
le = LabelEncoder()
target = le.fit_transform(target)

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 모델 생성
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(le.classes_), activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 모델 저장
model.save('hand_gesture_model')
