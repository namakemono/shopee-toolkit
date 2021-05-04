import numpy as np
from typing import List
import tensorflow as tf
import cv2
import shopee

def get_image_embeddings(df, image_size, weights_name:str):
    if weights_name in ["effnet_b3_weights_keras", "effnet_b3_swav_weights_keras"]:
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


