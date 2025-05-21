import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
import torch
import mediapipe as mp
from transformer_sign_model import TransformerSignModel

SEQ_LEN = 50
INPUT_DIM = 126
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clase_a_palabra = {
    f"{i:04d}": nombre for i, nombre in enumerate([
        "Gracias", "Buenos Días", "Buenas Tardes", "Buenas Noches", "Seña", "Nombre", "Trabajar", "Comer", "Vivir", "Poco",
        "Familia", "Personas", "Mujer", "Hombre", "Niño", "Niña", "Abuelo", "Tío", "Hermano", "Hambre",
        "Feliz", "Contento", "Triste", "Aburrido", "Bien", "Mal", "¿Cómo estás?", "Más o menos", "Sentir", "Jucipñoso",
        "Hola", "Adiós", "Por favor", "Con gusto", "Bienvenido", "Perdón", "Permiso", "Nunca", "Yo", "Tú",
        "Ustedes", "¿Qué?", "¿Cuándo?", "¿Dónde?", "¿Cómo?", "¿Por qué?", "¿Quién?", "Diferente", "Todos", "Mucho"
    ])
}

# Cargar modelo
model = TransformerSignModel(INPUT_DIM, seq_len=SEQ_LEN).to(DEVICE)
model.load_state_dict(torch.load("modelo_lsc_transformer.pth", map_location=DEVICE))
model.eval()

mp_hands = mp.solutions.hands

class SignRecognizer(VideoTransformerBase):
    def __init__(self):
        self.buffer_frames = []
        self.palabra = "Esperando..."
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        left_hand = np.zeros(63)
        right_hand = np.zeros(63)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                if hand_handedness.classification[0].label == 'Left':
                    left_hand = hand_array
                else:
                    right_hand = hand_array

        frame_data = np.concatenate([left_hand, right_hand])
        self.buffer_frames.append(frame_data)

        if len(self.buffer_frames) > SEQ_LEN:
            self.buffer_frames.pop(0)

        if len(self.buffer_frames) == SEQ_LEN:
            sequence = np.stack(self.buffer_frames).astype(np.float32)
            sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(sequence_tensor)
                pred = torch.argmax(output, dim=1).item()
                self.palabra = clase_a_palabra.get(f"{pred:04d}", "Desconocido")

        cv2.putText(image, f"Seña: {self.palabra}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image

# Interfaz web
st.set_page_config(page_title="Reconocimiento de LSC", layout="centered")
st.title("🤟 Reconocimiento de Lengua de Señas Colombiana (LSC)")
st.markdown("Utiliza tu cámara desde celular o PC. Modelo basado en Transformer.")

webrtc_streamer(
    key="lsc-stream",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=SignRecognizer,
    media_stream_constraints={"video": True, "audio": False}
)
