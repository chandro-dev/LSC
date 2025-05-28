import pandas as pd
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def balance_dataset(df: pd.DataFrame, repeticiones_por_clase=4):
    """
    Para cada clase en la columna 'N', genera múltiples entradas repetidas (1 por repetición esperada).
    Esto no cambia los frames, pero permite buscar múltiples archivos CSV por clase en el __getitem__ del Dataset.
    """
    nuevas_filas = []
    for _, row in df.iterrows():
        for _ in range(repeticiones_por_clase):
            nuevas_filas.append(row.copy())
    df_balanceado = pd.DataFrame(nuevas_filas)
    return df_balanceado
