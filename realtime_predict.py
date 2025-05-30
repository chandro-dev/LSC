import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from collections import deque, Counter
import mediapipe as mp
import random
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# === Configuraciones ===
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
           'X', 'Y', 'Z', 'del', 'nothing', 'space']

INITIAL_TARGETS = ['L', 'W', 'B', 'I']  # Letras iniciales específicas

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("resnet18_asl_model.pth", map_location="cpu"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def pad_to_square(img):
    h, w, _ = img.shape
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

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
        self.used_letters = set()
        self.initial_targets = INITIAL_TARGETS.copy()
        self.target_letter = self.get_new_target_letter()
        self.score = 0
        self.mode_game = tk.BooleanVar(value=True)
        self.predictions = deque(maxlen=60)
        self.last_letter = None
        self.last_written_letter = None

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
        tk.Button(control_frame, text="Siguiente Letra", command=self.next_letter).grid(row=0, column=2, padx=10)
        tk.Button(control_frame, text="Salir", command=self.quit).grid(row=0, column=3, padx=10)

    def get_new_target_letter(self):
        if self.initial_targets:
            next_initial = self.initial_targets.pop(0)
            self.used_letters.add(next_initial)
            return next_initial
        available = [l for l in CLASSES[:26] if l not in self.used_letters]
        if not available:
            messagebox.showinfo("Juego terminado", "¡Has completado todas las letras!")
            self.used_letters.clear()
            self.initial_targets = INITIAL_TARGETS.copy()
            return self.get_new_target_letter()
        return random.choice(available)

    def next_letter(self):
        self.target_letter = self.get_new_target_letter()
        self.target_label.config(text=f"{self.target_letter}")

    def clear_word(self):
        self.word_var.set("")
        self.last_written_letter = None

    def quit(self):
        self.running = False
        self.root.quit()

    def run_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Evitar parpadeo con DirectShow
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir el búfer para baja latencia
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la cámara")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            letter = "..."

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    x_min = max(int(min(x_coords) * w) - 20, 0)
                    x_max = min(int(max(x_coords) * w) + 20, w)
                    y_min = max(int(min(y_coords) * h) - 20, 0)
                    y_max = min(int(max(y_coords) * h) + 20, h)

                    roi = frame[y_min:y_max, x_min:x_max]
                    if roi.size == 0:
                        continue
                    roi = pad_to_square(roi)
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    roi_pil = Image.fromarray(roi_rgb)
                    input_tensor = transform(roi_pil).unsqueeze(0)

                    with torch.no_grad():
                        output = model(input_tensor)
                        pred_index = output.argmax(1).item()
                        pred_letter = CLASSES[pred_index]

                        self.predictions.append(pred_index)

                    if len(self.predictions) == self.predictions.maxlen:
                        most_common = Counter(self.predictions).most_common(1)[0][0]
                        letter = CLASSES[most_common]
                        self.letter_var.set(letter)

                        if letter == "space":
                            if self.last_written_letter != "space":
                                self.word_var.set(self.word_var.get() + " ")
                                self.last_written_letter = "space"
                        elif letter == "del":
                            current_word = self.word_var.get()
                            self.word_var.set(current_word[:-1])
                            self.last_written_letter = None
                        elif letter != "nothing" and letter != self.last_written_letter:
                            self.word_var.set(self.word_var.get() + letter)
                            self.last_written_letter = letter

                        if self.mode_game.get() and letter == self.target_letter:
                            self.score += 1
                            self.score_label.config(text=f"Puntaje: {self.score}")
                            self.used_letters.add(self.target_letter)
                            messagebox.showinfo("Correcto", f"¡Correcto! Era la letra {self.target_letter}")
                            self.next_letter()

                        self.predictions.clear()

            frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_display).resize((800, 480), Image.NEAREST)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()
