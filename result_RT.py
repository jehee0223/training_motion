import cv2
import torch
import mediapipe as mp
from torch.utils.data import DataLoader
from tensor_split import MyDataset  # MyDataset 클래스 가져오기
import LSTM as model
import numpy as np

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.3)
draw_line = [[11, 13], [13, 15], [12, 14], [14, 16], [23, 25], [25, 27], [24, 26], [26, 28], [11, 12], [11, 23],
             [23, 24], [12, 24]]

# LSTM 모델 초기화
model.init_model()
model.net.load_state_dict(torch.load('model/final_model_30_2.pth', map_location=torch.device('cpu')))
model.net.eval()

# 출력 창 크기 설정
OUTPUT_WIDTH, OUTPUT_HEIGHT = 630, 1120
sequence_length = 30
interval = 2

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def alert(draw_line_dic, prediction):
    warning_message = None

    if prediction == 'Squat':
        left_shoulder = draw_line_dic[11]
        right_shoulder = draw_line_dic[12]
        left_ankle = draw_line_dic[27]
        right_ankle = draw_line_dic[28]

        if left_ankle[0] < left_shoulder[0] or right_ankle[0] > right_shoulder[0]:
            warning_message = "Your feet are too close."

    elif prediction == 'SLR':
        elbow = draw_line_dic[14]
        shoulder = draw_line_dic[12]
        hip = draw_line_dic[24]

        angle = calculate_angle(elbow, shoulder, hip)
        if angle > 90:
            warning_message = "Your shoulders are too high."


    elif prediction == 'SP':
        elbow = draw_line_dic[14]
        shoulder = draw_line_dic[12]
        angle = calculate_angle(elbow, shoulder, (shoulder[0], shoulder[1] - 1))  # 수직 정렬 확인

        if angle > 110:
            warning_message = "Your shoulders are too low."

        print(angle)

    return warning_message

def count(draw_line_dic, prediction, prev_squat_position, prev_slr_position, prev_sp_position, squat_count, slr_count, sp_count):
    squat_position = prev_squat_position
    slr_position = prev_slr_position
    sp_position = prev_sp_position

    if prediction == 'Squat':
        hip = draw_line_dic[24]
        knee = draw_line_dic[26]
        ankle = draw_line_dic[28]

        angle = calculate_angle(hip, knee, ankle)

        if angle < 150:
            squat_position = "down"
        elif angle > 170:
            squat_position = "up"

        if prev_squat_position == "down" and squat_position == "up":
            squat_count += 1

    if prediction == 'SLR':
        elbow = draw_line_dic[14]
        shoulder = draw_line_dic[12]
        hip = draw_line_dic[24]

        angle = calculate_angle(elbow, shoulder, hip)

        if angle < 30:
            slr_position = "down"
        elif angle > 70:
            slr_position = "up"

        if prev_slr_position == "up" and slr_position == "down":
            slr_count += 1

    if prediction == 'SP':
        elbow = draw_line_dic[14]
        shoulder = draw_line_dic[12]
        hip = draw_line_dic[24]

        angle = calculate_angle(elbow, shoulder, hip)

        if angle < 60:
            sp_position = "down"
        elif angle > 120:
            sp_position = "up"

        if prev_sp_position == "up" and sp_position == "down":
            sp_count += 1

    return squat_position, slr_position, sp_position, squat_count, slr_count, sp_count

def process_real_time(interval, sequence_length):
    cap = cv2.VideoCapture(0)  # 0번 카메라 장치 사용

    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    frame_count = 0
    xy_list_list = []
    prediction = "Unknown"
    squat_count = 0
    slr_count = 0
    sp_count = 0
    prev_squat_position = None
    prev_slr_position = None
    prev_sp_position = None
    warning_message = ''

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 원본 해상도 가져오기
        original_height, original_width = frame.shape[:2]

        # 중앙 크롭 영역 계산
        start_x = max((original_width - OUTPUT_WIDTH) // 2, 0)
        start_y = max((original_height - OUTPUT_HEIGHT) // 2, 0)
        end_x = start_x + OUTPUT_WIDTH
        end_y = start_y + OUTPUT_HEIGHT

        # 프레임 중앙 크롭
        frame = frame[start_y:end_y, start_x:end_x]

        draw_line_dic = {}

        if frame_count % interval == 0:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                xy_list = []

                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if idx in [p[0] for p in draw_line] or idx in [p[1] for p in draw_line]:
                        x, y = int(landmark.x * OUTPUT_WIDTH), int(landmark.y * OUTPUT_HEIGHT)
                        xy_list.append(landmark.x)
                        xy_list.append(landmark.y)
                        draw_line_dic[idx] = (x, y)

                xy_list_list.append(xy_list)

                prev_squat_position, prev_slr_position, prev_sp_position, squat_count, slr_count, sp_count = count(
                    draw_line_dic, prediction, prev_squat_position, prev_slr_position, prev_sp_position, squat_count,
                    slr_count, sp_count
                )

                warning_message = alert(draw_line_dic, prediction)

                if len(xy_list_list) == sequence_length:
                    dataset = [{'key': 0, 'value': xy_list_list}]
                    dataset = MyDataset(dataset)
                    dataloader = DataLoader(dataset, batch_size=1)

                    for data, _ in dataloader:
                        data = data.to(model.device)
                        with torch.no_grad():
                            result = model.net(data)
                            _, pred_class = torch.max(result, 1)
                            prediction = ["Squat", "SLR", "SP", "waiting"][pred_class.item()]

                    xy_list_list = xy_list_list[int(sequence_length / 2):]

        for line in draw_line:
            if line[0] in draw_line_dic and line[1] in draw_line_dic:
                pt1 = draw_line_dic[line[0]]
                pt2 = draw_line_dic[line[1]]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        cv2.putText(frame, f"Prediction: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if prediction == 'Squat':
            cv2.putText(frame, f"Squat Count: {squat_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction == 'SLR':
            cv2.putText(frame, f"SLR Count: {slr_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction == 'SP':
            cv2.putText(frame, f"SP Count: {sp_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if warning_message:
            cv2.putText(frame, warning_message, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Real-Time Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# 실시간 프로세스 실행
process_real_time(interval, sequence_length)

