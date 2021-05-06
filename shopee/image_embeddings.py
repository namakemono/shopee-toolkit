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
    if entry_id in ["effnet_b3_arcface_pytorch"]:
        entry = shopee.registry.get_entry_by_id(entry_id)
        get_image_embeddings = shopee.image_embeddings_pytorch.get_image_embeddings(
            df              = df,
            image_size      = entry["image_size"]
        )
    else:
        return shopee.image_embeddings_keras.get_image_embeddings(
            entry_id        = entry_id, 
            df              = df,
        )

