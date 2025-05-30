import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from collections import deque, Counter
import mediapipe as mp
import random
import time
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# === Configuraciones ===
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
           'X', 'Y', 'Z', 'del', 'nothing', 'space']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("resnet18_asl_model.pth", map_location="cpu"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

class ASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Señas - ASL")
        self.root.geometry("1000x700")
        self.root.configure(bg="white")

        self.canvas = tk.Label(self.root)
        self.canvas.pack(pady=10)

        self.letter_var = tk.StringVar(value="...")
        self.word_var = tk.StringVar(value="")
        self.target_letter = random.choice(CLASSES[:26])
        self.score = 0
        self.mode_game = tk.BooleanVar(value=True)
        self.predictions = deque(maxlen=45)  # más muestras para mayor precisión
        self.start_time = time.time()
        self.frame_counter = 0
        self.frame_interval = 5

        self.build_ui()
        self.running = True
        self.video_thread = threading.Thread(target=self.run_camera)
        self.video_thread.daemon = True
        self.video_thread.start()

    def build_ui(self):
        info_frame = tk.Frame(self.root, bg="white")
        info_frame.pack()

        tk.Label(info_frame, text="Letra Detectada:", bg="white", font=("Arial", 14)).grid(row=0, column=0)
        tk.Label(info_frame, textvariable=self.letter_var, fg="green", bg="white", font=("Arial", 18, "bold")).grid(row=0, column=1)

        tk.Label(info_frame, text="Palabra:", bg="white", font=("Arial", 14)).grid(row=1, column=0)
        tk.Label(info_frame, textvariable=self.word_var, bg="white", font=("Arial", 14)).grid(row=1, column=1)

        self.target_label = tk.Label(info_frame, text=f"{self.target_letter}", bg="white", font=("Arial", 14))
        tk.Label(info_frame, text="Objetivo:", bg="white", font=("Arial", 14)).grid(row=2, column=0)
        self.target_label.grid(row=2, column=1)

        self.score_label = tk.Label(info_frame, text=f"Puntaje: {self.score}", bg="white", font=("Arial", 14))
        self.score_label.grid(row=3, column=0, columnspan=2)

        control_frame = tk.Frame(self.root, bg="white")
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="Limpiar", command=self.clear_word).grid(row=0, column=0, padx=10)
        tk.Checkbutton(control_frame, text="Modo Juego", variable=self.mode_game, bg="white").grid(row=0, column=1, padx=10)
        tk.Button(control_frame, text="Salir", command=self.quit).grid(row=0, column=2, padx=10)

    def clear_word(self):
        self.word_var.set("")

    def quit(self):
        self.running = False
        self.root.quit()

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_counter += 1
            if self.frame_counter % self.frame_interval != 0:
                continue

            x1, y1, x2, y2 = 100, 100, 324, 324
            roi = frame[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            result = hands.process(roi_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                roi_pil = transforms.ToPILImage()(roi_rgb)
                input_tensor = transform(roi_pil).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred_index = output.argmax(1).item()
                    self.predictions.append(pred_index)

                if len(self.predictions) == self.predictions.maxlen:
                    most_common = Counter(self.predictions).most_common(1)[0][0]
                    letter = CLASSES[most_common]
                    self.letter_var.set(letter)

                    if letter == "space":
                        self.word_var.set(self.word_var.get() + " ")
                    elif letter == "del":
                        self.word_var.set(self.word_var.get()[:-1])
                    elif letter != "nothing":
                        self.word_var.set(self.word_var.get() + letter)

                    if self.mode_game.get() and letter == self.target_letter:
                        self.score += 1
                        self.target_letter = random.choice(CLASSES[:26])
                        self.target_label.config(text=f"{self.target_letter}")
                        self.score_label.config(text=f"Puntaje: {self.score}")

                    self.predictions.clear()

            # Mostrar imagen
            frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_display)
            img = img.resize((800, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()