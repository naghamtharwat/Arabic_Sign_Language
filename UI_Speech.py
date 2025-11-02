
import tkinter as tk
from tkinter import messagebox
import serial
import numpy as np
import joblib
from gtts import gTTS
import tempfile
import os
import threading
import time

# ========== Serial Settings ==========
PORT = "COM25" 
BAUD = 115200

# ========== Load Models ==========
letters_model = joblib.load("rf_letters.pkl")
numbers_model = joblib.load("rf_numbers.pkl")
Words_model   = joblib.load("rf_Words.pkl")
letters_scaler = joblib.load("scaler_letters.pkl")
numbers_scaler = joblib.load("scaler_numbers.pkl")
Words_scaler   = joblib.load("scaler_Words.pkl")

# ========== Columns ==========
COLUMNS = ['flex1','flex2','flex3','flex4','flex5',
           'ax1','ay1','az1','gx1','gy1','gz1',
           'flex6','flex7','flex8','flex9','flex10',
           'ax2','ay2','az2','gx2','gy2','gz2']

# ========== Helper: Speech Arabic ==========
def speak(text):
    try:
        tts = gTTS(text=str(text), lang="ar")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
        tts.save(tmp_path)
        os.system(f'start "" "{tmp_path}"')
        def remove_tmp_file(path):
            time.sleep(10)
            try:
                os.remove(path)
            except:
                pass
        threading.Thread(target=remove_tmp_file, args=(tmp_path,), daemon=True).start()
    except Exception as e:
        print("TTS error:", e)

# ========== Helper: Numbers to Arabic ==========
def number_to_arabic(num):
    mapping = {
        "10": "١٠", "1": "١", "2": "٢", "3": "٣", "4": "٤",
        "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩", "0": "٠"
    }
    return mapping.get(str(num), str(num))

# ========== Helper: Read Serial ==========
def read_from_serial():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        line = ser.readline().decode(errors='ignore').strip()
        ser.close()
        if line:
            parts = [float(x) for x in line.split(",") if x.strip()]
            if len(parts) == len(COLUMNS):
                return np.array(parts).reshape(1, -1)
        return None
    except Exception as e:
        messagebox.showerror("Serial Error", f"Cannot read from {PORT}:\n{e}")
        return None

# ========== Main Prediction Function ==========
def predict(mode):
    def run():
        set_status("Reading data...", "blue")
        X = read_from_serial()
        if X is None:
            set_status("No data received", "red")
            return

        try:
            if mode == "letters":
                X_scaled = letters_scaler.transform(X)
                pred = letters_model.predict(X_scaled)[0]
                result_text = f"الحرف المتوقع: {pred}"
                speak(pred)

            elif mode == "numbers":
                X_scaled = numbers_scaler.transform(X)
                pred_num = numbers_model.predict(X_scaled)[0]
                pred_num = int(round(pred_num))
                arabic_num = number_to_arabic(pred_num)
                result_text = f"الرقم المتوقع: {arabic_num}"
                speak(arabic_num)

            elif mode == "words":
                X_scaled = Words_scaler.transform(X)
                pred_word = Words_model.predict(X_scaled)[0]
                result_text = f"الكلمة المتوقعة: {pred_word}"
                speak(pred_word)

            else:
                result_text = "وضع غير معروف"

            result_label.config(text=result_text)
            set_status("Prediction Success!", "green")

        except Exception as e:
            set_status("Prediction error", "red")
            messagebox.showerror("Prediction Error", str(e))

    threading.Thread(target=run).start()

# ========== Status Function ==========
def set_status(text, color):
    status_label.config(text=text, fg=color)
    status_label.update()

# ========== UI ==========
root = tk.Tk()
root.title("Smart Glove Predictor")
root.geometry("520x530")
root.configure(bg="#1b1f2a")

# Fonts and colors
font_title = ("Segoe UI", 18, "bold")
font_label = ("Segoe UI", 11)
font_button = ("Segoe UI", 11, "bold")
bg_main = "#1b1f2a"
accent_blue = "#6c8ef0"
accent_green = "#3ac569"
accent_red = "#e74c3c"
accent_yellow = "#f1c40f"

# ===== Title =====
tk.Label(root, text="🖐 Smart Glove Predictor", font=font_title,
         fg="#ffffff", bg=bg_main).pack(pady=(25, 10))
tk.Label(root, text=f"Connected to Arabic Sign Language UI", font=font_label,
         fg="#c9c9c9", bg=bg_main).pack()

# ===== Buttons =====
btn_frame = tk.Frame(root, bg=bg_main)
btn_frame.pack(pady=40)

def hover_on(e): e.widget.config(bg="#728cff")
def hover_off(e): e.widget.config(bg=accent_blue)
def hover_on_green(e): e.widget.config(bg="#41da74")
def hover_off_green(e): e.widget.config(bg=accent_green)
def hover_on_yellow(e): e.widget.config(bg="#f4d03f")
def hover_off_yellow(e): e.widget.config(bg=accent_yellow)
def hover_exit_on(e): e.widget.config(bg="#ff665a")
def hover_exit_off(e): e.widget.config(bg=accent_red)

btn_letters = tk.Button(btn_frame, text="Predict Letter", font=font_button,
                        bg=accent_blue, fg="white", width=16, height=2,
                        relief="flat", bd=0, command=lambda: predict("letters"))
btn_letters.grid(row=0, column=0, padx=10)
btn_letters.bind("<Enter>", hover_on)
btn_letters.bind("<Leave>", hover_off)

btn_numbers = tk.Button(btn_frame, text="Predict Number", font=font_button,
                        bg=accent_green, fg="white", width=16, height=2,
                        relief="flat", bd=0, command=lambda: predict("numbers"))
btn_numbers.grid(row=0, column=1, padx=10)
btn_numbers.bind("<Enter>", hover_on_green)
btn_numbers.bind("<Leave>", hover_off_green)

# === New Word Prediction Button ===
btn_words = tk.Button(btn_frame, text="Predict Word", font=font_button,
                      bg=accent_yellow, fg="black", width=16, height=2,
                      relief="flat", bd=0, command=lambda: predict("words"))
btn_words.grid(row=1, column=0, columnspan=2, pady=10)
btn_words.bind("<Enter>", hover_on_yellow)
btn_words.bind("<Leave>", hover_off_yellow)

# ===== Result =====
result_label = tk.Label(root, text="Prediction will appear here",
                        font=("Segoe UI", 14, "bold"), fg="#86b7ff", bg=bg_main)
result_label.pack(pady=40)

# ===== Status =====
status_label = tk.Label(root, text="", font=("Segoe UI", 10, "italic"),
                        fg="gray", bg=bg_main)
status_label.pack()

# ===== Exit Button =====
btn_exit = tk.Button(root, text="Exit", font=font_button,
                     bg=accent_red, fg="white", width=10, height=1,
                     relief="flat", command=root.destroy)
btn_exit.pack(pady=20)
btn_exit.bind("<Enter>", hover_exit_on)
btn_exit.bind("<Leave>", hover_exit_off)

root.mainloop()
