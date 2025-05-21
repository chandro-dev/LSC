
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class LSC50Dataset(Dataset):
    def __init__(self, base_dir, seq_len=50, use_body=True, use_face=True, use_hands=True, augment=False):
        self.seq_len = seq_len
        self.use_body = use_body
        self.use_face = use_face
        self.use_hands = use_hands
        self.augment = augment

        self.left_hand_dir = os.path.join(base_dir, "HANDS_LANDMARKS/LEFT_HAND_LANDMARKS")
        self.right_hand_dir = os.path.join(base_dir, "HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS")
        self.body_dir = os.path.join(base_dir, "BODY_LANDMARKS")
        self.face_dir = os.path.join(base_dir, "FACE_LANDMARKS")

        self.files = sorted(os.listdir(self.left_hand_dir))
        self.label_map = {f"{i:04d}": i for i in range(50)}

    def _load_landmarks(self, path):
        df = pd.read_csv(path)
        
        # Quitar columnas sin nombre (como 'Unnamed: 0')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Convertir a float por seguridad
        return df.to_numpy(dtype=np.float32)


    def normalize_landmarks(self, landmarks):
        # landmarks: (T, N*3)
        T, D = landmarks.shape
        if D % 3 != 0:
            raise ValueError("El número de columnas debe ser múltiplo de 3 (x, y, z por punto)")

        landmarks_reshaped = landmarks.reshape(T, -1, 3)  # (T, N, 3)
        center = landmarks_reshaped[:, 0:1, :]            # (T, 1, 3), usando el primer landmark como centro
        normalized = landmarks_reshaped - center          # broadcasting (T, N, 3)
        return normalized.reshape(T, -1)                  # volver a (T, N*3)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        key = file.split('_')[0]
        label = self.label_map[key]
    

        seqs = []

        if self.use_hands:
            left = self._load_landmarks(os.path.join(self.left_hand_dir, file))
            right = self._load_landmarks(os.path.join(self.right_hand_dir, file))
            seqs.append(left)
            seqs.append(right)

        if self.use_body:
            body = self._load_landmarks(os.path.join(self.body_dir, file))
            seqs.append(body)

        if self.use_face:
            face = self._load_landmarks(os.path.join(self.face_dir, file))
            seqs.append(face)

        min_frames = min(s.shape[0] for s in seqs)
        seqs = [s[:min_frames] for s in seqs]
        seqs = [self.normalize_landmarks(s) for s in seqs]

        X = np.concatenate(seqs, axis=1)

        if X.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - X.shape[0], X.shape[1]))
            X = np.vstack((X, pad))
        else:
            X = X[:self.seq_len]

        if self.augment:
            X += np.random.normal(0, 0.01, X.shape)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(label, dtype=torch.long)