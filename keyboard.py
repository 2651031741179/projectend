import tkinter as tk
from tkinter import messagebox
import os
import numpy as np
import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import shutil

# FONT
FONT_PATH = r"THSarabunNew.ttf"

# Parameters
DATA_PATH = os.path.join('MP_Data')

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Data Check and Utility Functions ---

def get_existing_actions():
    """ดึงรายการคำศัพท์ (ชื่อโฟลเดอร์) ที่มีอยู่ในระบบ"""
    if not os.path.exists(DATA_PATH):
        return []
    return sorted([
        d for d in os.listdir(DATA_PATH) 
        if os.path.isdir(os.path.join(DATA_PATH, d)) and not d.startswith('.')
    ])

def get_action_status(action, no_sequences, sequence_length):
    """ตรวจสอบสถานะความสมบูรณ์ของคำศัพท์แต่ละคำ"""
    action_path = os.path.join(DATA_PATH, action)
    
    # 1. นับจำนวน Sequences ที่เก็บได้ (ชื่อโฟลเดอร์เป็นตัวเลข)
    collected_sequences = sorted([
        d for d in os.listdir(action_path) if d.isdigit()
    ], key=int)
    num_sequences = len(collected_sequences)
    
    # ตรวจสอบว่าเก็บ Sequences ครบหรือไม่
    if num_sequences < no_sequences:
        return f"⚠️ ไม่ครบ ({num_sequences}/{no_sequences} ครั้ง)"

    # 2. ตรวจสอบว่าทุก Sequence มี Frame ครบหรือไม่
    for seq_folder in collected_sequences:
        seq_path = os.path.join(action_path, seq_folder)
        # นับจำนวนไฟล์ .npy (Keypoints) ในแต่ละ Sequence
        num_frames = len([f for f in os.listdir(seq_path) if f.endswith('.npy')])
        if num_frames < sequence_length:
            # Sequences นี้มี Frame ไม่ครบ
            return f"⚠️ ไม่ครบ (วิดีโอ {seq_folder} ภาพนิ่ง: {num_frames}/{sequence_length})"
    
    # ผ่านทุกเงื่อนไข
    return f"✅ สมบูรณ์ ({num_sequences} ครั้ง)"

def update_existing_actions_display(listbox, count_label, no_sequences, sequence_length):
    """อัพเดท Listbox พร้อมแสดงสถานะความสมบูรณ์ตามค่าที่ตั้งไว้"""
    listbox.delete(0, tk.END)
    existing_actions = get_existing_actions()
    
    if existing_actions:
        for action in existing_actions:
            status = get_action_status(action, no_sequences, sequence_length)
            listbox.insert(tk.END, f"{action} | {status}")
        
        count_label.config(text=f"มีคำอยู่ในระบบ: {len(existing_actions)} คำ (อ้างอิง: ต้องเก็บ {no_sequences} ครั้ง, ความยาว {sequence_length} ภาพนิ่ง)")
    else:
        listbox.insert(tk.END, "ไม่มีข้อมูลคำศัพท์")
        count_label.config(text="มีคำอยู่ในระบบ: 0 คำ")

def delete_selected_action(listbox):
    """ฟังก์ชันสำหรับลบคำศัพท์ที่เลือกออกจากระบบ (ลบโฟลเดอร์ใน MP_Data)"""
    selected_indices = listbox.curselection()
    if not selected_indices:
        messagebox.showwarning("แจ้งเตือน", "กรุณาเลือกคำศัพท์ที่ต้องการลบในรายการด้านบนก่อน")
        return

    # ดึงชื่อคำศัพท์ที่เลือก (ตัดส่วนสถานะออก)
    selected_item = listbox.get(selected_indices[0])
    action_to_delete = selected_item.split(" | ")[0].strip()

    confirm = messagebox.askyesno(
        "ยืนยันการลบ", 
        f"คุณต้องการลบคำว่า '{action_to_delete}' ออกจากระบบถาวรหรือไม่?\n(ข้อมูลทั้งหมดจะถูกลบและกู้คืนไม่ได้)"
    )

    if confirm:
        try:
            action_path = os.path.join(DATA_PATH, action_to_delete)
            shutil.rmtree(action_path)
            messagebox.showinfo("สำเร็จ", f"คำว่า '{action_to_delete}' ถูกลบออกจากระบบแล้ว")
            
            # ลบคำที่ถูกลบออกจากรายการ actions ที่รอเก็บด้วย
            if action_to_delete in actions:
                actions.remove(action_to_delete)
                status.set(f"คำที่รอเก็บ: {', '.join(actions)}")

            # รีเฟรชรายการคำศัพท์ใน GUI
            refresh_display()
        except Exception as e:
            messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่สามารถลบคำศัพท์ได้: {e}")


# --- MediaPipe and OpenCV Core Functions (ไม่เปลี่ยนแปลง) ---

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

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
    left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left, right])

def put_thai_text(image, text, position, font_size=32, color=(0, 255, 0)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# --- GUI 1: Settings, Action List, and Word Input ---

def add_word():
    """เพิ่มคำศัพท์ใหม่เข้ารายการที่รอเก็บ"""
    word = entry.get().strip()
    if word:
        if word in get_existing_actions():
             messagebox.showwarning("แจ้งเตือน", f"คำว่า '{word}' มีอยู่แล้วในระบบ! ระบบจะเริ่มเก็บข้อมูลต่อจากที่ขาดไป")
             # ไม่ต้อง return ให้เพิ่มใน actions เพื่อนำไปเริ่มเก็บต่อ
        
        if word not in actions:
            actions.append(word)
            entry.delete(0, tk.END)
            status.set(f"คำที่รอเก็บ: {', '.join(actions)}")
        else:
            messagebox.showwarning("แจ้งเตือน", f"คำว่า '{word}' ถูกเพิ่มในรายการรอเก็บแล้ว")
    else:
        messagebox.showwarning("แจ้งเตือน", "กรุณากรอกคำก่อน")

def start_collection_and_capture():
    """ตรวจสอบการตั้งค่าและเริ่มกระบวนการเก็บข้อมูลวิดีโอ"""
    if not actions:
        messagebox.showwarning("แจ้งเตือน", "กรุณาเพิ่มคำที่ต้องการเก็บข้อมูลก่อน")
        return

    try:
        # ดึงค่าที่ผู้ใช้ป้อน
        no_sequences_val = int(sequences_var.get())
        sequence_length_val = int(length_var.get())
        det_conf_val = float(det_conf_var.get())
        track_conf_val = float(track_conf_var.get())
    except ValueError:
        messagebox.showerror("ค่าผิดพลาด", "กรุณาตรวจสอบค่าตัวเลขทั้งหมดในช่อง 'ตั้งค่า'")
        return

    window.destroy()  # ปิด GUI
    start_capture(actions, no_sequences_val, sequence_length_val, det_conf_val, track_conf_val)

# --- Core Capture Logic (ปรับปรุงการแจ้งเตือนขณะข้าม) ---

def start_capture(actions, no_sequences, sequence_length, min_detection_confidence, min_tracking_confidence):
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as holistic:
        for action in actions:
            print(f"--- เริ่มประมวลผลคำ: {action} ---")
            os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)
            
            all_skipped = True # ตัวแปรสำหรับตรวจสอบว่าข้ามทุก Sequence หรือไม่

            for sequence in range(no_sequences):
                # *** Logic สำหรับการ "เก็บต่อ" หรือ "ข้าม" ที่ผู้ใช้ถามถึง ***
                seq_path = os.path.join(DATA_PATH, action, str(sequence))
                
                # 1. ตรวจสอบว่า Sequence นี้เก็บข้อมูลครบแล้วหรือไม่
                if os.path.exists(seq_path):
                    num_frames_collected = len([f for f in os.listdir(seq_path) if f.endswith('.npy')])
                    if num_frames_collected == sequence_length:
                        print(f"ข้าม: {action} วิดีโอที่ {sequence} เพราะเก็บครบ {sequence_length} ภาพนิ่งแล้ว")
                        continue # ข้ามไปยัง Sequence ถัดไปได้เลย
                    elif num_frames_collected > 0:
                         print(f"บันทึกทับ: {action} วิดีโอที่ {sequence} เพราะมีข้อมูลไม่ครบ ({num_frames_collected}/{sequence_length})")
                # ***************************************************************

                all_skipped = False
                
                # กด F เพื่อเริ่ม
                waiting = True
                while waiting:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = put_thai_text(frame, f"กด F เพื่อเริ่มเก็บ: {action} วิดีโอที่ {sequence} / {no_sequences - 1}", (30, 30), color=(0, 0, 255))
                    cv2.imshow('OpenCV Feed', frame)
                    if cv2.waitKey(1) & 0xFF == ord('f'):
                        for i in range(5, 0, -1):
                            ret, frame = cap.read()
                            h, w, _ = frame.shape
                            frame = put_thai_text(frame, str(i), (w // 2 - 50, h // 2 - 50), font_size=120, color=(0, 255, 255))
                            cv2.imshow('OpenCV Feed', frame)
                            cv2.waitKey(1000)
                        waiting = False

                # สร้างโฟลเดอร์สำหรับ Sequence นี้ (จะสร้าง/ทับโฟลเดอร์เดิม)
                os.makedirs(seq_path, exist_ok=True)
                
                # เก็บ Keypoints
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret: break
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    # แสดงสถานะการเก็บข้อมูล
                    image = put_thai_text(image, f'เก็บ: {action} วิดีโอ {sequence}/{no_sequences - 1} ภาพนิ่ง: {frame_num}/{sequence_length - 1}', 
                                            (10, 10), font_size=24, color=(0, 0, 255))
                    cv2.imshow('OpenCV Feed', image)

                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(seq_path, f"{frame_num}.npy"), keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            
            if all_skipped:
                print(f"--- คำว่า '{action}' ข้ามการเก็บทั้งหมด เพราะสมบูรณ์แล้ว ---")


    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("เสร็จสิ้น", "เก็บข้อมูลเสร็จสมบูรณ์!")


# --- GUI Setup ---
actions = []
window = tk.Tk()
window.title("ระบบเตรียมข้อมูลภาษามือ")
window.geometry("800x850") 

# --- Section 1: Data Collection Settings ---
tk.Label(window, text="--- ⚙️ ขั้นตอนที่ 1: ตั้งค่าการเก็บข้อมูล (ปรับค่าเพื่อควบคุมความเร็ว/ปริมาณ) ---", 
         font=("TH Sarabun New", 18, "bold"), bg="lightgrey").pack(pady=10, fill='x')

frame_settings = tk.Frame(window)
frame_settings.pack(padx=20, pady=10)

# ตัวแปรสำหรับค่าใหม่
sequences_var = tk.StringVar(value="30")
length_var = tk.StringVar(value="30")
det_conf_var = tk.StringVar(value="0.5")
track_conf_var = tk.StringVar(value="0.5")

# ใช้ป้ายกำกับภาษาคน
labels = [
    "จำนวนครั้งที่ต้องทำท่า (วิดีโอตัวอย่าง):", 
    "ความยาวท่าทาง (จำนวนภาพนิ่งต่อครั้ง):",
    "ความเชื่อมั่นการตรวจจับ (Detection Confidence):", 
    "ความเชื่อมั่นการติดตาม (Tracking Confidence):"
]
vars_list = [sequences_var, length_var, det_conf_var, track_conf_var]

for i, label_text in enumerate(labels):
    tk.Label(frame_settings, text=label_text, font=("TH Sarabun New", 14)).grid(row=i, column=0, sticky='w', padx=10, pady=2)
    tk.Entry(frame_settings, textvariable=vars_list[i], width=10, font=("TH Sarabun New", 14)).grid(row=i, column=1, padx=10, pady=2)


# --- Section 2: Existing Actions ---
tk.Label(window, text="--- ✅ ขั้นตอนที่ 2: ตรวจสอบและจัดการคำศัพท์ที่มีในระบบ ---", 
         font=("TH Sarabun New", 18, "bold"), bg="lightgrey").pack(pady=10, fill='x')

frame_actions = tk.Frame(window)
frame_actions.pack(padx=20, pady=10, fill='x')

count_label = tk.Label(frame_actions, text="กำลังโหลด...", font=("TH Sarabun New", 14), fg="blue")
count_label.pack()

listbox_actions = tk.Listbox(frame_actions, height=8, font=("TH Sarabun New", 14), width=50, exportselection=0)
listbox_actions.pack(pady=5, fill='x')

# ฟังก์ชัน Lambda สำหรับปุ่มรีเฟรช
def refresh_display():
    try:
        no_seq = int(sequences_var.get())
        seq_len = int(length_var.get())
        update_existing_actions_display(listbox_actions, count_label, no_seq, seq_len)
    except ValueError:
        messagebox.showwarning("แจ้งเตือน", "กรุณากรอกค่า จำนวนครั้ง/ความยาว เป็นตัวเลขก่อนรีเฟรช")

# ปุ่มจัดการคำศัพท์
frame_control = tk.Frame(window)
frame_control.pack(pady=5)

tk.Button(frame_control, text="🔄 รีเฟรชรายการคำ", command=refresh_display,
          font=("TH Sarabun New", 14), bg='skyblue').grid(row=0, column=0, padx=10)

tk.Button(frame_control, text="🗑️ ลบคำศัพท์ที่เลือก (ลบถาวร)", command=lambda: delete_selected_action(listbox_actions),
          font=("TH Sarabun New", 14), bg='red', fg='white').grid(row=0, column=1, padx=10)

# รีเฟรชครั้งแรกเมื่อโปรแกรมเปิด
refresh_display() 


# --- Section 3: Word Input and Start Button ---
tk.Label(window, text="--- 📹 ขั้นตอนที่ 3: กรอกคำศัพท์ใหม่และเริ่มเก็บข้อมูล ---", 
         font=("TH Sarabun New", 18, "bold"), bg="lightgrey").pack(pady=10, fill='x')

tk.Label(window, text="กรอกคำศัพท์ใหม่ทีละคำ (เช่น สวัสดี, ขอบคุณ):", font=("TH Sarabun New", 14)).pack(pady=5)
entry = tk.Entry(window, font=("TH Sarabun New", 18), width=30)
entry.pack(pady=5)

tk.Button(window, text="➕ เพิ่มคำในรายการรอเก็บ", font=("TH Sarabun New", 16), command=add_word, bg='lightgreen').pack(pady=5)

status = tk.StringVar(value="คำที่รอเก็บ: ไม่มี")
tk.Label(window, textvariable=status, font=("TH Sarabun New", 16), fg="blue").pack(pady=10)

tk.Button(window, text="🎬 เริ่มเก็บข้อมูลวิดีโอ", 
          font=("TH Sarabun New", 18), bg="green", fg="white", command=start_collection_and_capture).pack(pady=20)


window.mainloop()