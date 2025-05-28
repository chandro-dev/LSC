import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 75

# Configurar logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProporcionLoader:
    @staticmethod
    def cargar_proporciones(vol_dir):
        proporciones = {}
        for file in os.listdir(vol_dir):
            match = re.search(r"Volunteer_(\d+)", file)
            if not match:
                continue
            vol_id = int(match.group(1))
            with open(os.path.join(vol_dir, file), "r") as f:
                lines = f.read().splitlines()
                props = {}
                for line in lines:
                    if ":" in line and "Hz" not in line:
                        try:
                            k, v = line.split(":")
                            k = k.strip().lower()
                            v = float(v.strip().replace("cm", ""))
                            props[k] = v
                        except ValueError:
                            continue
                proporciones[vol_id] = props
        return proporciones


class LandmarkLoader:
    def __init__(self, base_dir, proporciones):
        self.base_dir = base_dir
        self.proporciones = proporciones
        self.body_dir = os.path.join(base_dir, 'BODY_LANDMARKS')
        self.face_dir = os.path.join(base_dir, 'FACE_LANDMARKS')
        self.left_hand_dir = os.path.join(base_dir, 'HANDS_LANDMARKS', 'LEFT_HAND_LANDMARKS')
        self.right_hand_dir = os.path.join(base_dir, 'HANDS_LANDMARKS', 'RIGHT_HAND_LANDMARKS')

    def load_csv(self, filepath):
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                return df.values
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
                return None
        return None

    def get_voluntario_id(self, filename):
        match = re.match(r"\d{4}_(\d{4})_", filename)
        return int(match.group(1)) if match else None

    def normalizar_por_voluntario(self, data, vol_id):
        escala = self.proporciones.get(vol_id, {}).get("back (from coccyx to cervical vertebrae c7)", 1)
        return data / escala if escala else data

    def load_all_landmarks(self, filename, use_body=True, use_face=True, use_hands=True):
        vol_id = self.get_voluntario_id(filename)
        data_parts = []

        if use_body:
            body_data = self.load_csv(os.path.join(self.body_dir, filename))
            if body_data is not None:
                data_parts.append(body_data)

        if use_face:
            face_data = self.load_csv(os.path.join(self.face_dir, filename))
            if face_data is not None:
                data_parts.append(face_data)

        if use_hands:
            left = self.load_csv(os.path.join(self.left_hand_dir, filename))
            right = self.load_csv(os.path.join(self.right_hand_dir, filename))
            hands = None
            if left is not None and right is not None:
                min_len = min(left.shape[0], right.shape[0])
                hands = np.concatenate([left[:min_len], right[:min_len]], axis=1)
            elif left is not None:
                hands = left
            elif right is not None:
                hands = right
            if hands is not None:
                data_parts.append(hands)

        if not data_parts:
            raise ValueError(f"No landmark data found for any modality in {filename}")

        min_len = min(part.shape[0] for part in data_parts)
        data_parts_aligned = [part[:min_len] for part in data_parts]
        all_data = np.concatenate(data_parts_aligned, axis=1)
        return self.normalizar_por_voluntario(all_data, vol_id)


class SignLanguageDataset(Dataset):
    def __init__(self, loader, label_table, seq_len=75):
        self.loader = loader
        self.label_table = label_table
        self.seq_len = seq_len
        
        self.analyze_class_distribution()
        self.available_files = self._find_available_files()

    def analyze_class_distribution(self):
        class_counts = Counter(self.label_table['N'])
        logger.info(f"Distribución de clases: {dict(class_counts)}")
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        if max_count / min_count > 10:
            logger.warning(f"¡DESBALANCE EXTREMO! Min: {min_count}, Max: {max_count}")

    def _find_available_files(self):
        available = {}
        for directory in [self.loader.body_dir, self.loader.face_dir, 
                          self.loader.left_hand_dir, self.loader.right_hand_dir]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.csv'):
                        match = re.match(r"(\d{4})_(\d{4})_(\d{4})\.csv", file)
                        if match:
                            sign_id = int(match.group(1))
                            available.setdefault(sign_id, []).append(file)
        logger.info(f"Archivos encontrados para {len(available)} signos diferentes")
        return available

    def __len__(self):
        return len(self.label_table)

    def __getitem__(self, idx):
        row = self.label_table.iloc[idx]
        sign_id = int(row['N'])
        start, end = int(row["start_frame"]), int(row["end_frame"])

        if sign_id not in self.available_files:
            logger.warning(f"No se encontraron archivos para el signo {sign_id}")
            dummy_data = np.zeros((self.seq_len, 100))
            return torch.tensor(dummy_data, dtype=torch.float32), torch.tensor(sign_id)

        # Selección aleatoria de una de las repeticiones disponibles (3 CSVs por clase)
        filename = np.random.choice(self.available_files[sign_id])

        try:
            landmarks = self.loader.load_all_landmarks(filename)
        except Exception as e:
            logger.error(f"Error cargando {filename}: {e}")
            dummy_data = np.zeros((self.seq_len, 100))
            return torch.tensor(dummy_data, dtype=torch.float32), torch.tensor(sign_id)

        max_frames = landmarks.shape[0]
        if start >= max_frames or start < 0:
            start = 0
        if end >= max_frames or end < start:
            end = max_frames - 1
        if start > end:
            start = 0
            end = max_frames - 1

        segment = landmarks[start:end+1]

        if segment.shape[0] >= self.seq_len:
            if segment.shape[0] > self.seq_len:
                start_idx = np.random.randint(0, segment.shape[0] - self.seq_len + 1)
                segment = segment[start_idx:start_idx + self.seq_len]
            else:
                segment = segment[:self.seq_len]
        else:
            pad = np.zeros((self.seq_len - segment.shape[0], segment.shape[1]))
            segment = np.vstack([segment, pad])

        # Augmentación fuerte (activada siempre para mejorar generalización)
        #segment += np.random.normal(0, 0.01, size=segment.shape)  # Ruido aleatorio
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            segment *= factor  # Escalado global

        x = torch.tensor(segment, dtype=torch.float32)
        y = torch.tensor(sign_id, dtype=torch.long)
        return x, y
