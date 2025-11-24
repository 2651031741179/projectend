import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk
import pyttsx3 # นำเข้า pyttsx3

# ======== ให้ผู้ใช้เลือกโมเดลผ่านหน้าต่าง ========
Tk().withdraw()
model_path = filedialog.askopenfilename(
    title='เลือกไฟล์โมเดล .h5',
    filetypes=[("H5 files", "*.h5")]
)
if not model_path:
    print("❌ ไม่ได้เลือกโมเดล")
    exit()

model = load_model(model_path)

# ======== อ่านคำทั้งหมดจากโฟลเดอร์ MP_Data ========
DATA_PATH = 'MP_Data'
actions = np.array(sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]))
print("🟢 คำที่โหลด:", actions)

# กำหนดสีสำหรับการแสดงผล (สุ่มตามจำนวนคำ)
colors = [(np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)) for _ in actions]

# Initialize pyttsx3 Engine (ตั้งค่า Offline TTS Engine)
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    print(f"❌ ไม่สามารถเริ่มต้น pyttsx3 engine ได้: {e}")
    tts_engine = None


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
    return np.concatenate([pose, left_hand, right_hand]).flatten()

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def speak_offline(text):
    """แปลงข้อความเป็นเสียงและเล่นเสียงแบบ Offline โดยใช้ pyttsx3"""
    global tts_engine
    if tts_engine:
        tts_engine.stop() 
        tts_engine.say(text)
        tts_engine.runAndWait() 


# ตรวจจับการเคลื่อนไหว
sequence = []
sentence = []
predictions = []
threshold = 0.3

# เปิดกล้อง (ใช้ CAP_DSHOW เพื่อเพิ่มความเสถียรบน Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # เพิ่ม cv2.CAP_DSHOW

# *** ตรวจสอบการเปิดกล้อง ***
if not cap.isOpened():
    print("❌ FATAL ERROR: ไม่สามารถเข้าถึงกล้องได้ โปรดตรวจสอบว่ากล้องถูกเสียบอยู่ และไม่ได้ถูกใช้โดยโปรแกรมอื่น")
    exit()
# **************************

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ ไม่สามารถอ่านเฟรมจากกล้องได้")
            break

        # image = cv2.resize(frame, (640, 480)) # สามารถ uncomment บรรทัดนี้เพื่อเพิ่มความเร็ว

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):  
                if res[np.argmax(res)] > threshold:  
                    predicted_word = actions[np.argmax(res)]
                    
                    if len(sentence) > 0:  
                        if predicted_word != sentence[-1]:  
                            sentence.append(predicted_word)
                            speak_offline(predicted_word) 
                    else:
                        sentence.append(predicted_word)
                        speak_offline(predicted_word) 

            if len(sentence) > 5:  
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)

        # ย้ายตำแหน่งข้อความไปด้านล่างเพื่อให้เห็นชัดขึ้น
        # 640x480 (กว้าง x สูง)
        cv2.rectangle(image, (0, 440), (640, 480), (245, 117, 16), -1) # กล่องพื้นหลังด้านล่าง
        cv2.putText(image, ' '.join(sentence), (3, 470), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()