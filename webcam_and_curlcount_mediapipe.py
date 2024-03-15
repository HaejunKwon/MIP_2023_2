from ultralytics import YOLO
import cv2
from mediapipe import solutions as mp

# MediaPipe Pose 모델 초기화
mp_pose = mp.pose.Pose()
cap = cv2.VideoCapture(0)  # 기본 카메라 사용, 다른 카메라를 사용하려면 값을 변경하세요.

# Colab에서 학습한 모델을 다운로드한 경로로 수정
model_path = "C:/Users/82109/Desktop/MIP/best.pt"

# Load the YOLO model
model = YOLO(model_path)

# pull, push 동작 반복 횟수 초기화
pull_count = 0
push_count = 0

while True:
    ret, frame = cap.read()  # 비디오에서 프레임 읽기
    
    # Mediapipe를 사용하여 key point 예측 수행
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(image)

    # key point 표시
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # key point의 좌표값 받아오기 (wrist, elbow, shoulder)
            if landmark.visibility > 0.5:  # 시각적으로 인식 가능한 key point만 표시
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # 원으로 key point 표시

    # YOLO 모델을 사용하여 예측 수행
    results = model.predict(source=frame, show=True, conf=0.5)  # frame을 직접 전달

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