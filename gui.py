import tkinter as tk
import subprocess
import threading

def run_script(script_name):
    def target():
        subprocess.Popen(["python", script_name])
    threading.Thread(target=target, daemon=True).start()

root = tk.Tk()
root.title("ระบบแปลภาษามือ")
root.geometry("400x300")

tk.Label(root, text="เมนูหลัก", font=("TH Sarabun New", 24)).pack(pady=20)

tk.Button(root, text="🟩 เตรียมคำและเก็บข้อมูล", command=lambda: run_script("keyboard.py"),
          font=("TH Sarabun New", 16), width=30, bg='green', fg='white').pack(pady=10)

tk.Button(root, text="🟨 เทรนโมเดล", command=lambda: run_script("trian.py"),
          font=("TH Sarabun New", 16), width=30, bg='gold').pack(pady=10)

tk.Button(root, text="🟦 ใช้งานโมเดล", command=lambda: run_script("เสียง.py"),
          font=("TH Sarabun New", 16), width=30, bg='skyblue').pack(pady=10)

tk.Button(root, text="🟥 ออก", command=root.destroy,
          font=("TH Sarabun New", 16), width=30, bg='red', fg='white').pack(pady=10)

root.mainloop()
