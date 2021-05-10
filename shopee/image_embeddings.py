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
    entry = shopee.registry.get_entry_by_id(entry_id)
    if entry.model_type == "keras-origin":
        return shopee.image_embeddings_keras.get_image_embeddings(
            entry_id        = entry_id,
            df              = df,
        )
    elif entry.model_type == "pytorch-arcface":
        return shopee.image_embeddings_pytorch.get_image_embeddings(
            df              = df,
            image_size      = entry.image_size,
            weights_filepath= entry.weights_filepath,
        )
    else:
        raise ValueError(f"Undefined model type: {entry.model_type}")
