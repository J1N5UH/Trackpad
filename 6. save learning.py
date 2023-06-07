import tensorflow as tf

# 학습 이력 파일 경로 지정
history_path = "여기에 경로"

# 학습 이력 불러오기
with open(history_path, "rb") as f:
    history = pickle.load(f)

# 모델 생성
model = create_model()

# 모델의 가중치 및 매개변수 설정
model.set_weights(history["weights"])
model.optimizer.set_weights(history["optimizer_weights"])
