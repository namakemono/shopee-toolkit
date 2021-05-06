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
    df:pd.DataFrame,
    image_size:int,
    weights_name:str,
):
    if weights_name in [
        "effnet_b3_weights_keras",
        "effnet_b7_weights_keras",
        "effnet_b3_swav_weights_keras",
        "effnet_b0_ns_weights_keras",
        "effnet_b3_ns_weights_keras",
    ]:
        return shopee.image_embeddings_keras.get_image_embeddings(
            df=df,
            image_size=image_size,
            weights_name=weights_name,
        )
    elif weights_name in ["effnet-b3-arcface-pytorch"]:
        get_image_embeddings = shopee.image_embeddings_pytorch.get_image_embeddings(
            df,
            image_size
        )
    else:
        raise ValueError(f"Undefined engine name: {weights_name}")


