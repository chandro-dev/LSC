import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from transformer_sign_model import TransformerSignModel

# Configuración
SEQ_LEN = 50
INPUT_DIM = 126  # 2 manos * 21 puntos * 3 coords
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = TransformerSignModel(input_dim=INPUT_DIM, seq_len=SEQ_LEN).to(DEVICE)
model.load_state_dict(torch.load("modelo_lsc_transformer.pth", map_location=DEVICE))
model.eval()

# Diccionario de clases
clase_a_palabra = {
    f"{i:04d}": nombre for i, nombre in enumerate([
        "Gracias", "Buenos Días", "Buenas Tardes", "Buenas Noches", "Seña", "Nombre", "Trabajar", "Comer", "Vivir", "Poco",
        "Familia", "Personas", "Mujer", "Hombre", "Niño", "Niña", "Abuelo", "Tío", "Hermano", "Hambre",
        "Feliz", "Contento", "Triste", "Aburrido", "Bien", "Mal", "¿Cómo estás?", "Más o menos", "Sentir", "Jucipñoso",
        "Hola", "Adiós", "Por favor", "Con gusto", "Bienvenido", "Perdón", "Permiso", "Nunca", "Yo", "Tú",
        "Ustedes", "¿Qué?", "¿Cuándo?", "¿Dónde?", "¿Cómo?", "¿Por qué?", "¿Quién?", "Diferente", "Todos", "Mucho"
    ])
}

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cv2.namedWindow("Reconocimiento de Señas - LSC", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reconocimiento de Señas - LSC", 960, 720)

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ No se pudo abrir la cámara.")

    buffer_frames = []
    palabra_actual = "Esperando..."

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al capturar el frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Extraer landmarks de ambas manos
        left_hand = np.zeros(63)
        right_hand = np.zeros(63)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                if hand_handedness.classification[0].label == 'Left':
                    left_hand = hand_array
                else:
                    right_hand = hand_array
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_data = np.concatenate([left_hand, right_hand])
        buffer_frames.append(frame_data)
        if len(buffer_frames) > SEQ_LEN:
            buffer_frames.pop(0)

        cv2.putText(frame, f"Frames: {len(buffer_frames)}/{SEQ_LEN}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if len(buffer_frames) == SEQ_LEN:
            sequence = np.stack(buffer_frames).astype(np.float32)
            sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(sequence_tensor)
                pred = torch.argmax(output, dim=1).item()
                palabra_actual = clase_a_palabra.get(f"{pred:04d}", "Desconocido")

        cv2.putText(frame, f"Seña: {palabra_actual}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Reconocimiento de Señas - LSC", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            print("👋 Finalizando...")
            break

    cap.release()
    cv2.destroyAllWindows()
