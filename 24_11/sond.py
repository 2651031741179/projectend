import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk
from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
from playsound import playsound
import uuid

# --- ฟอนต์ภาษาไทย ---
FONT_PATH = r"THSarabunNew.ttf"

# ==========================================================
# ========== โหลดไฟล์โมเดล (Hardcoded Path) ===============
# ==========================================================
model_path = 'eatrice.h5'
if not os.path.exists(model_path):
    print(f"❌ FATAL ERROR: ไม่พบไฟล์โมเดล '{model_path}'")
    exit()

model = load_model(model_path)
print(f"🟢 โหลดโมเดลสำเร็จ: {model_path}")

# ======== โหลดคำจากโฟลเดอร์ MP_Data ========
DATA_PATH = 'MP_Data'
if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: ไม่พบโฟลเดอร์ '{DATA_PATH}'")
    exit()

actions = np.array(sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]))
if len(actions) == 0:
    print("❌ ERROR: ไม่พบโฟลเดอร์คำศัพท์ใน MP_Data")
    exit()
print("🟢 คำที่โหลด:", actions)

# --- [แก้ไข 1] สร้างสีเผื่อไว้เยอะๆ (เช่น 100 สี) ป้องกัน Index Error ถ้ารายการคำไม่เท่ากับโมเดล ---
colors = [(np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)) for _ in range(100)]


# ==========================================================
# ========== ฟังก์ชัน TTS (เสียงภาษาไทย) ===================
# ==========================================================
def speak_thai(text):
    """แปลงข้อความเป็นเสียงไทย + เล่นทันที"""
    try:
        filename = f"tts_{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang='th')
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"⚠️ TTS ERROR: {e}")


# ==========================================================
# ========== ฟังก์ชัน Mediapipe & Utils ====================
# ==========================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def put_thai_text(image, text, position, font_path=FONT_PATH, font_size=32, color=(255, 255, 255)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # print("❌ Font not found, using default font") # ปิด print รกหน้าจอ
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

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

def prob_viz(res, actions, input_frame, colors, is_active):
    output_frame = input_frame.copy()
    if is_active:
        for num, prob in enumerate(res):
            # --- [แก้ไข 2] เช็คว่า num ไม่เกินจำนวน actions และ colors ที่มีอยู่ ---
            if num < len(actions) and num < len(colors):
                cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
                output_frame = put_thai_text(output_frame, actions[num], (0, 65 + num * 40), font_size=20)
    return output_frame


# ==========================================================
# =================== Logic จับท่าทาง ======================
# ==========================================================
sequence = []
sentence = []
predictions = []
threshold = 0.7
current_frame_count = 0
CLEAR_TIMEOUT_FRAMES = 60
last_detection_frame = 0
res = [] # เริ่มต้นเป็น list ว่าง

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ ไม่สามารถเข้าถึงกล้องได้")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ ไม่สามารถอ่านเฟรมจากกล้องได้")
            break

        current_frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        is_new_detection = False
        is_prediction_run = False

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            
            # --- [แก้ไข 3] ป้องกัน Error หาก res มีขนาดใหญ่กว่า actions ---
            if len(res) > len(actions):
                # ถ้าโมเดลทำนายออกมาได้ class เกินกว่า actions ที่เรามี ให้ตัด res ทิ้ง
                res_to_use = res[:len(actions)]
            else:
                res_to_use = res

            predictions.append(np.argmax(res_to_use))
            is_prediction_run = True

            if np.unique(predictions[-10:])[0] == np.argmax(res_to_use):
                if res_to_use[np.argmax(res_to_use)] > threshold:
                    
                    # ตรวจสอบ index ก่อนเรียกใช้ actions
                    idx = np.argmax(res_to_use)
                    if idx < len(actions):
                        predicted_word = actions[idx]

                        if len(sentence) > 0:
                            if predicted_word != sentence[-1]:
                                sentence.append(predicted_word)
                                is_new_detection = True
                                speak_thai(predicted_word)
                        else:
                            sentence.append(predicted_word)
                            is_new_detection = True
                            speak_thai(predicted_word)

            if len(sentence) > 5:
                sentence = sentence[-5:]

        # --- ล้างประโยคเมื่อหยุด ---
        if is_new_detection:
            last_detection_frame = current_frame_count
            is_viz_active = True
        elif current_frame_count - last_detection_frame < CLEAR_TIMEOUT_FRAMES and len(sentence) > 0:
            is_viz_active = True
        elif current_frame_count - last_detection_frame >= CLEAR_TIMEOUT_FRAMES and len(sentence) > 0:
            sentence = []
            is_viz_active = False
            print("💡 ล้างประโยคเพราะหยุดทำท่าทาง")
        else:
            is_viz_active = False

        if is_prediction_run or current_frame_count < 30:
            # ส่ง res เข้าไป (ถ้ายังไม่มีค่าให้ข้าม)
            if len(res) > 0:
                image = prob_viz(res, actions, image, colors, is_viz_active)

        cv2.rectangle(image, (0, 440), (640, 480), (245, 117, 16), -1)

        display_text = ' '.join(sentence) if sentence else "เริ่มทำท่าทาง..."
        image = put_thai_text(image, display_text, (3, 445), font_size=30)

        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()