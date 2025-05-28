import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from collections import deque
from PIL import Image


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
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(num_classes=25).to(device)
model.load_state_dict(torch.load("C:/Users/PC/Documents/repos/ProyectoLenguajeSenas/cnn_sign_mnist_model.pth", map_location=device))
model.eval()

# === Transformaciones ===
transform = transforms.Compose([
    transforms.Resize((28, 28)),
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
            xmin = int(min(x_coords) * w) - 20
            xmax = int(max(x_coords) * w) + 20
            ymin = int(min(y_coords) * h) - 20
            ymax = int(max(y_coords) * h) + 20


            # Recortar y predecir
            roi = frame[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (28, 28))
            roi = cv2.bitwise_not(roi)  # Invertir colores: fondo negro, mano blanca            
            if roi.size > 0:
                try:
                    letter = predict_letter(Image.fromarray(roi))
                    letter_buffer.append(letter)
                    # Solo añadir si se repite varias veces seguidas
                    if len(set(letter_buffer)) == 1:
                        word += letter
                        letter_buffer.clear()
                except:
                    pass

    # Mostrar resultados
    cv2.putText(frame, f"Palabra: {word}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Sign Language Letter Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        word = ""  # Limpiar palabra

cap.release()

cv2.destroyAllWindows()
