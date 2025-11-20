import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# --- GUI Functionality ---
def start_training():
    model_name = model_name_var.get()
    try:
        epochs = int(epochs_var.get())
        batch_size = int(batch_var.get())
    except ValueError:
        messagebox.showerror("ค่าผิดพลาด", "กรุณากรอก Epochs และ Batch เป็นตัวเลข")
        return

    if not model_name:
        messagebox.showerror("กรุณาตั้งชื่อโมเดล", "กรุณากรอกชื่อโมเดลก่อนเริ่มเทรน")
        return

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
                    res = np.load(npy_file)
                else:
                    res = np.zeros((75, 3))
                window.append(res.flatten())
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)

    if len(X) == 0:
        messagebox.showerror("ไม่มีข้อมูล", "ไม่พบข้อมูลในโฟลเดอร์ MP_Data")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    status_label.config(text="✅ เริ่มเทรนโมเดล...")
    root.update()

    # Define model
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, feature_dim)),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    progress_bar.start()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stop])
    progress_bar.stop()

    model.save(f'{model_name}.h5')
    status_label.config(text=f"✅ เทรนเสร็จสิ้น บันทึกเป็น {model_name}.h5")
    messagebox.showinfo("สำเร็จ", f"โมเดลถูกบันทึกเป็น {model_name}.h5")

# --- GUI Setup ---
root = tk.Tk()
root.title("เทรนโมเดลแปลภาษามือ")
root.geometry("500x350")

model_name_var = tk.StringVar()
epochs_var = tk.StringVar(value="300")
batch_var = tk.StringVar(value="32")

tk.Label(root, text="ชื่อโมเดล:").pack(pady=5)
tk.Entry(root, textvariable=model_name_var, width=30).pack()

tk.Label(root, text="จำนวน Epochs:").pack(pady=5)
tk.Entry(root, textvariable=epochs_var, width=10).pack()

tk.Label(root, text="Batch Size:").pack(pady=5)
tk.Entry(root, textvariable=batch_var, width=10).pack()

tk.Button(root, text="▶️ เริ่มเทรน", command=start_training, bg="green", fg="white", width=20).pack(pady=15)

progress_bar = ttk.Progressbar(root, mode="indeterminate")
progress_bar.pack(fill='x', padx=30, pady=5)

status_label = tk.Label(root, text="รอเริ่มเทรน...")
status_label.pack(pady=10)

root.mainloop()
