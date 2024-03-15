from ultralytics import YOLO
import cv2
import numpy as np

# Colab에서 학습한 모델을 다운로드한 경로로 수정
model_path = "C:/Users/82109/Desktop/MIP/best.pt"
# Load the YOLO model
model = YOLO(model_path)


# 전처리 함수 정의
def preprocess_image(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러 적용하여 노이즈 제거
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 적절한 임계값을 사용하여 이미지 이진화
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # BGR에서 RGB로 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image

# pull, push 동작 반복 횟수 초기화
pull_count = 0
push_count = 0

# 실시간 비디오 처리를 위한 웹캠 비디오 스트림 열기
cap = cv2.VideoCapture(0)  # 기본 카메라 사용, 다른 카메라를 사용하려면 값을 변경하세요.

while True:
    ret, frame = cap.read()  # 비디오에서 프레임 읽기

    preprocessed_frame=preprocess_image(frame)

    # YOLO 모델을 사용하여 예측 수행
    results = model.predict(source=preprocessed_frame, show=True, conf=0.5)  # frame을 직접 전달

    # 수정된 코드
    if results and 'xyxy' in results[0]:
        for label, conf, box in results[0]['xyxy'][:, :3]:
            # 여기에 원하는 동작 추가
            print(f"Label: {label}, Confidence: {conf}, Box: {box}")

            # 'pull' 또는 'push'에 대한 동작 판별
            if label == 'Pull':
                pull_count += 1
            elif label == 'Push':
                push_count += 1

    else:
        print("No detections")

    # repeat count 계산
    repeat_count = max(pull_count + push_count - 1, 0)

    # 화면에 repeat count 표시
    cv2.putText(frame, f'Repeat Count: {repeat_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 프레임 화면에 표시
    cv2.imshow("Webcam", frame)

    # 'q'를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체 해제 및 창 닫기
cap.release()

# 열린 모든 창 닫기
cv2.destroyAllWindows()
