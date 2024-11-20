import cv2
import mediapipe as mp
from tqdm import tqdm
import random
import os
import pickle

def mp_to_lstm_pose(video_path, interval):
    mp_pose = mp.solutions.pose
    attention_dot = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    xy_list_list = []

    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.3)
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        cnt = 0
        while True:
            ret, img = cap.read()
            if cnt == interval and ret == True:
                cnt = 0
                xy_list = []
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not results.pose_landmarks:
                    continue
                landmarks = results.pose_landmarks.landmark

                # 관심 랜드마크(attention_dot)만 추출
                for idx in attention_dot:
                    xy_list.append(landmarks[idx].x)
                    xy_list.append(landmarks[idx].y)

                xy_list_list.append(xy_list)
            elif ret == False:
                break
            cnt += 1
    cap.release()
    cv2.destroyAllWindows()
    return xy_list_list

# train_video 폴더에 있는 모든 파일 처리
video_folder = './drive/MyDrive/late_dataset'
video_list = os.listdir(video_folder)

# 데이터셋 생성
dataset = []
sequence_length = 30  # 한 시퀀스에 포함될 프레임 수
interval = 2

for video_name in tqdm(video_list):
    # 파일 이름에서 라벨 추출 (slr -> 1, squat -> 0)
    if 'squat' in video_name.lower():
        label = 0
    elif 'slr' in video_name.lower():
        label = 1
    elif 'sp' in video_name.lower():
        label = 2
    elif 'standing' in video_name.lower():
        label = 3
    else:
        print(f"Unknown label for file: {video_name}, skipping...")
        continue

    # 비디오 처리
    video_path = f"{video_folder}/{video_name}"
    xy_list_list = mp_to_lstm_pose(video_path, interval)

    # 슬라이딩 윈도우 방식으로 시퀀스 추출
    for start_idx in range(0, len(xy_list_list) - sequence_length + 1, int(sequence_length/2)):
        sequence = xy_list_list[start_idx:start_idx + sequence_length]
        if len(sequence) == sequence_length:
            dataset.append({'key': label, 'value': sequence})

# 데이터셋 셔플
random.shuffle(dataset)

# 데이터셋 저장
output_path = './drive/MyDrive/final_dataset_30_2.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)

print(f"Dataset created with {len(dataset)} samples and saved to {output_path}")
