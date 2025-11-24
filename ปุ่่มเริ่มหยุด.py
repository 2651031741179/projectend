<<<<<<< HEAD
import os
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import time

# Path to Thai font
FONT_PATH = r"/Users/pangpp/Desktop/แอปพลิเคชันแปลงภาษามือเป็นข้อความ/code/Beforetest/Font/THSarabunNew/THSarabunNew.ttf"

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['สวัสดี', 'ขอบคุณ', 'ใช่'])

# Number of sequences and sequence length
no_sequences = 30
sequence_length = 30

# Folder start
start_folder = 0

# Ensure directories exist
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand])

def put_thai_text(image, text, position, font_path=FONT_PATH, font_size=32, color=(0, 255, 0)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("❌ Font not found, using default font")
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Start capturing video
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):

            # รอให้ผู้ใช้กด F ก่อนเริ่มเก็บข้อมูล
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break

                frame = put_thai_text(frame, f"กด F เพื่อเริ่มเก็บข้อมูล: {action} วิดีโอที่ {sequence}", (30, 30), font_size=32, color=(0, 0, 255))
                cv2.imshow('OpenCV Feed', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('f'):
                    # แสดง countdown ใหญ่กลางจอ
                    for i in range(5, 0, -1):
                        ret, frame = cap.read()
                        h, w, _ = frame.shape
                        frame = put_thai_text(frame, str(i), (w // 2 - 50, h // 2 - 50), font_size=120, color=(0, 255, 255))
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(1000)
                    waiting = False

            seq_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(seq_path, exist_ok=True)

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    image = put_thai_text(image, 'เริ่มเก็บข้อมูล', (120, 200), font_size=40, color=(0, 255, 0))
                    image = put_thai_text(image, f'กำลังเก็บข้อมูล {action} วิดีโอที่ {sequence}', (15, 12), font_size=24, color=(0, 0, 255))
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    image = put_thai_text(image, f'กำลังเก็บข้อมูล {action} วิดีโอที่ {sequence}', (15, 12), font_size=24, color=(0, 0, 255))
                    cv2.imshow('OpenCV Feed', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(seq_path, f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
=======
import os
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import time

# Path to Thai font
FONT_PATH = r"/Users/pangpp/Desktop/แอปพลิเคชันแปลงภาษามือเป็นข้อความ/code/Beforetest/Font/THSarabunNew/THSarabunNew.ttf"

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['สวัสดี', 'ขอบคุณ', 'ใช่'])

# Number of sequences and sequence length
no_sequences = 30
sequence_length = 30

# Folder start
start_folder = 0

# Ensure directories exist
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand])

def put_thai_text(image, text, position, font_path=FONT_PATH, font_size=32, color=(0, 255, 0)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("❌ Font not found, using default font")
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Start capturing video
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):

            # รอให้ผู้ใช้กด F ก่อนเริ่มเก็บข้อมูล
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break

                frame = put_thai_text(frame, f"กด F เพื่อเริ่มเก็บข้อมูล: {action} วิดีโอที่ {sequence}", (30, 30), font_size=32, color=(0, 0, 255))
                cv2.imshow('OpenCV Feed', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('f'):
                    # แสดง countdown ใหญ่กลางจอ
                    for i in range(5, 0, -1):
                        ret, frame = cap.read()
                        h, w, _ = frame.shape
                        frame = put_thai_text(frame, str(i), (w // 2 - 50, h // 2 - 50), font_size=120, color=(0, 255, 255))
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(1000)
                    waiting = False

            seq_path = os.path.join(DATA_PATH, action, str(sequence))
            os.makedirs(seq_path, exist_ok=True)

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    image = put_thai_text(image, 'เริ่มเก็บข้อมูล', (120, 200), font_size=40, color=(0, 255, 0))
                    image = put_thai_text(image, f'กำลังเก็บข้อมูล {action} วิดีโอที่ {sequence}', (15, 12), font_size=24, color=(0, 0, 255))
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    image = put_thai_text(image, f'กำลังเก็บข้อมูล {action} วิดีโอที่ {sequence}', (15, 12), font_size=24, color=(0, 0, 255))
                    cv2.imshow('OpenCV Feed', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(seq_path, f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
>>>>>>> 1a5df781628a79c66062a559da563660ec133305
