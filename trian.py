import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras.callbacks import EarlyStopping, Callback 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
# *** แก้ไข: เพิ่มการนำเข้าที่ขาดหายไป ***
from sklearn.model_selection import train_test_split
# ****************************************
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# --- Custom Callback for GUI Logging ---
class GUILogger(Callback):
# ... (โค้ด GUILogger ไม่เปลี่ยนแปลง) ...
    """Callback ที่ส่งผลลัพธ์การฝึกไปยัง Text Widget ใน GUI"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.log_text = ""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # ดึงค่า Loss และ Accuracy
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_acc = logs.get('categorical_accuracy')
        val_acc = logs.get('val_categorical_accuracy')

        # จัดรูปแบบข้อความ
        log_line = f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n"
        
        # อัพเดท Text Widget
        self.text_widget.insert(tk.END, log_line)
        self.text_widget.see(tk.END) # เลื่อนลงไปด้านล่างสุด
        
        # ต้องเรียก root.update() เพื่อให้ Tkinter อัพเดทขณะที่ Keras รันอยู่
        self.text_widget.master.update()


# --- GUI Functionality ---
def start_training():
    model_name = model_name_var.get()
    try:
        epochs = int(epochs_var.get())
        batch_size = int(batch_var.get())
        
        # Get model structure and training parameters (ใช้ค่าจาก Dropdown)
        lstm1_units = int(lstm1_var.get())
        lstm2_units = int(lstm2_var.get())
        dense1_units = int(dense1_var.get())
        learning_rate = float(lr_var.get())
        patience = int(patience_var.get())
        
    except ValueError:
        messagebox.showerror("ค่าผิดพลาด", "กรุณาตรวจสอบค่าตัวเลขทั้งหมดในช่องกรอกข้อมูล")
        return

    if not model_name:
        messagebox.showerror("กรุณาตั้งชื่อโมเดล", "กรุณากรอกชื่อโมเดลก่อนเริ่มเทรน")
        return

    # Clear previous log
    log_text_widget.delete(1.0, tk.END)
    log_text_widget.insert(tk.END, "Loss/Accuracy Monitor Log:\n\n")

    status_label.config(text="🔄 กำลังโหลดข้อมูล...")
    root.update()

    # Read actions from folder
    DATA_PATH = 'MP_Data'
    actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])
    sequence_length = 30
    feature_dim = 75 * 3 
    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        for sequence in sorted([d for d in os.listdir(action_path) if d.isdigit()], key=int):
            sequence_path = os.path.join(action_path, sequence)
            window = []
            for frame_num in range(sequence_length):
                npy_file = os.path.join(sequence_path, f"{frame_num}.npy")
                if os.path.exists(npy_file):
                    try:
                        res = np.load(npy_file)
                    except ValueError:
                        res = np.zeros((75, 3)) 
                else:
                    res = np.zeros((75, 3))
                window.append(res.flatten())
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])

    X = np.array(sequences)
    
    if len(X) == 0:
        messagebox.showerror("ไม่มีข้อมูล", "ไม่พบข้อมูลที่สมบูรณ์ในโฟลเดอร์ MP_Data")
        return

    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    # *** บรรทัดนี้จะใช้ฟังก์ชันที่ถูกนำเข้าแล้ว ***
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    status_label.config(text="✅ เริ่มเทรนโมเดล...")
    root.update()
    
    # Initialize the custom logger callback
    gui_logger = GUILogger(log_text_widget)

    # Define model (ใช้ตัวแปรที่รับจากผู้ใช้)
    model = Sequential([
        LSTM(lstm1_units, return_sequences=True, activation='relu', input_shape=(sequence_length, feature_dim)), # ชั้น 1
        Dropout(0.2),
        LSTM(lstm2_units, return_sequences=True, activation='relu'), # ชั้น 2
        Dropout(0.2),
        LSTM(lstm1_units, return_sequences=False, activation='relu'), # ชั้น 3
        Dropout(0.2),
        Dense(dense1_units, activation='relu'), # ชั้น Dense 1
        Dense(int(dense1_units / 2), activation='relu'), # ชั้น Dense 2
        Dense(len(actions), activation='softmax')
    ])

    # Compile model (ใช้ตัวแคร Learning Rate ที่รับจากผู้ใช้)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Early Stopping (ใช้ตัวแปร Patience ที่รับจากผู้ใช้)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    progress_bar.start()
    
    # Add the custom logger to the callbacks list
    callbacks_list = [early_stop, gui_logger]
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callbacks_list)
    
    progress_bar.stop()

    model.save(f'{model_name}.h5')
    status_label.config(text=f"✅ เทรนเสร็จสิ้น บันทึกเป็น {model_name}.h5")
    messagebox.showinfo("สำเร็จ", f"โมเดลถูกบันทึกเป็น {model_name}.h5")

# --- GUI Setup ---
root = tk.Tk()
root.title("เทรนโมเดลแปลภาษามือ")
root.geometry("800x800") 

# Options for Units (เป็นเลขคู่ 16-128)
UNIT_OPTIONS = [16, 32, 64, 128]
# Options for Learning Rate
LR_OPTIONS = ["0.0001", "0.0002", "0.0005", "0.001"]

# Original Variables
model_name_var = tk.StringVar()
epochs_var = tk.StringVar(value="300")
batch_var = tk.StringVar(value="32")

# Model Architecture Variables (ใช้ Dropdown)
lstm1_var = tk.StringVar(value="64") 
lstm2_var = tk.StringVar(value="128")
dense1_var = tk.StringVar(value="64") 

# Training Parameter Variables
lr_var = tk.StringVar(value="0.0001")
patience_var = tk.StringVar(value="20")

# --- Top Frame for Settings ---
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

# Frame for Input 1
frame_input1 = tk.Frame(top_frame)
frame_input1.grid(row=0, column=0, padx=20, sticky='n')

# Frame for Input 2
frame_input2 = tk.Frame(top_frame)
frame_input2.grid(row=0, column=1, padx=20, sticky='n')

# --- Input Fields ---

# Basic Settings
tk.Label(frame_input1, text="--- 📝 ข้อมูลพื้นฐาน ---", font=("TH Sarabun New", 14, "bold")).pack(pady=5)
tk.Label(frame_input1, text="ชื่อไฟล์โมเดลที่ต้องการบันทึก:").pack(pady=2)
tk.Entry(frame_input1, textvariable=model_name_var, width=30).pack()
tk.Label(frame_input1, text="จำนวนรอบการฝึก (Epochs):").pack(pady=2)
tk.Entry(frame_input1, textvariable=epochs_var, width=10).pack()
tk.Label(frame_input1, text="ขนาดกลุ่มข้อมูล (Batch Size):").pack(pady=2)
tk.Entry(frame_input1, textvariable=batch_var, width=10).pack()
tk.Label(frame_input1, text="ความอดทน Early Stop (Patience):").pack(pady=2)
tk.Entry(frame_input1, textvariable=patience_var, width=10).pack()


# Model Architecture Inputs
tk.Label(frame_input2, text="--- 🧠 โครงสร้างโมเดล (Units) ---", font=("TH Sarabun New", 14, "bold")).pack(pady=5)

# LSTM 1 & 3
tk.Label(frame_input2, text="LSTM ชั้น 1 และ 3 Units:").pack(pady=2)
ttk.OptionMenu(frame_input2, lstm1_var, lstm1_var.get(), *UNIT_OPTIONS).pack()
tk.Label(frame_input2, text="*ยิ่งน้อย โมเดลยิ่งเร็ว แต่ความสามารถลดลง*").pack()

# LSTM 2
tk.Label(frame_input2, text="LSTM ชั้น 2 Units (ชั้นกลาง):").pack(pady=2)
ttk.OptionMenu(frame_input2, lstm2_var, lstm2_var.get(), *UNIT_OPTIONS).pack()
tk.Label(frame_input2, text="*ชั้นนี้สำคัญสุด ควรมีค่าสูงสุด*").pack()

# Dense 1
tk.Label(frame_input2, text="Dense ชั้น 1 Units:").pack(pady=2)
ttk.OptionMenu(frame_input2, dense1_var, dense1_var.get(), *UNIT_OPTIONS).pack()


# Training Parameter Inputs
tk.Label(frame_input2, text="--- ⚙️ อัตราการเรียนรู้ ---", font=("TH Sarabun New", 14, "bold")).pack(pady=10)

tk.Label(frame_input2, text="อัตราการเรียนรู้ (Learning Rate):").pack(pady=2)
ttk.OptionMenu(frame_input2, lr_var, lr_var.get(), *LR_OPTIONS).pack()
tk.Label(frame_input2, text="*ค่าสูงไป (เช่น 0.001) อาจทำให้โมเดลไม่แม่นยำ*").pack()


# --- Control and Status ---
tk.Button(root, text="▶️ เริ่มเทรนโมเดล", command=start_training, bg="green", fg="white", width=20).pack(pady=15)

progress_bar = ttk.Progressbar(root, mode="indeterminate")
progress_bar.pack(fill='x', padx=30, pady=5)

status_label = tk.Label(root, text="รอเริ่มเทรน...")
status_label.pack(pady=10)

# --- Monitor Log Display ---
tk.Label(root, text="📊 มอนิเตอร์ผลการเทรน (Loss/Accuracy ต่อ Epoch):", font=("TH Sarabun New", 14, "bold")).pack(pady=5)

log_text_widget = tk.Text(root, height=15, width=90, font=("Courier", 10), bg='black', fg='lime')
log_text_widget.pack(padx=20, pady=5)
log_text_widget.insert(tk.END, "รอเริ่มการฝึกโมเดล...\n")

root.mainloop()