import numpy as np
import pandas as pd
from typing import List
import tensorflow as tf
import cv2
import shopee

def load_image_embeddings(filepath:str) -> np.ndarray:
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        cols = [c for c in df.columns if "emb_" in c]
        return df[cols].values
    elif filepath.endswith(".npy"):
        return np.load(filepath)
    else:
        raise ValueError(f"Undefined extensions type: {filepath}")

def get_image_embeddings(
    entry_id: str,
    df: pd.DataFrame,
):
    return shopee.image_embeddings_keras.get_image_embeddings(
        entry_id        = entry_id, 
        df              = df,
    )

