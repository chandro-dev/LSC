import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from collections import deque
from PIL import Image
import time

# === Cargar modelo entrenado ===
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # Cambio aquí: 64 * 16 * 16 = 16384
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(num_classes=25).to(device)
model.load_state_dict(torch.load("C:/repos/LSC/cnn_sign_mnist_model.pth", map_location=device))
model.eval()

# === Transformaciones ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # CAMBIO: coincidir con entrenamiento
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# === MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# === Inicializar cámara
cap = cv2.VideoCapture(0)
word = ""
letter_buffer = deque(maxlen=10)  # Para suavizar predicciones
last_letter_time = time.time()
wait_seconds = 1.0  # tiempo de espera entre letras

def predict_letter(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
    return chr(pred + 65)  # A=65

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar mano
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener bbox alrededor de la mano
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            padding = 40  # Mayor recorte
            xmin = max(0, int(min(x_coords) * w) - padding)
            xmax = min(w, int(max(x_coords) * w) + padding)
            ymin = max(0, int(min(y_coords) * h) - padding)     
            ymax = min(h, int(max(y_coords) * h) + padding)


            # Recortar y predecir
   # Recortar y predecir
            roi = frame[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (64, 64))
            roi = cv2.bitwise_not(roi)  # Invertir colores

            if roi.size > 0:
                try:
                    # === Mostrar imagen de la mano antes de pasar al modelo ===
                    cv2.imshow("ROI", roi)

                    letter = predict_letter(Image.fromarray(roi))
                    letter_buffer.append(letter)

                    if len(set(letter_buffer)) == 1:
                        current_time = time.time()
                        if current_time - last_letter_time >= wait_seconds:
                            word += letter
                            letter_buffer.clear()
                            last_letter_time = current_time
                except:
                    pass
    # Mostrar resultados
    cv2.putText(frame, f"Palabra: {word}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Sign Language Letter Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        word = ""

cap.release()
cv2.destroyAllWindows()
