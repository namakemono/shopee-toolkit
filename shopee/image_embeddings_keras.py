import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import shopee

def to_images(filepath_list, image_size:int, preprocess_input=None):
    res = []
    for filepath in filepath_list:
        img = cv2.imread(filepath)
        img = cv2.resize(img, (image_size, image_size))
        res.append(img)
    X = np.asarray(res).astype(np.float32)
    if preprocess_input is not None:
        X = preprocess_input(X)
    return X

def predict_generator(df:pd.DataFrame, image_size:int, batch_size:int=8, preprocess_input=None):
    n = len(df)
    for i in range(0, n, batch_size):
        filepath_list = df.iloc[i:i+batch_size]["filepath"].tolist()
        images = to_images(filepath_list, image_size, preprocess_input)
        yield images

def get_efficientnet_backbone():
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights=None,
        input_shape=(None, None, 3)
    ) # can pretrain with ImageNet
    base_model.trainabe = True
    inputs = tf.keras.layers.Input((None, None, 3))
    h = base_model(inputs, training=True)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    backbone = tf.keras.models.Model(inputs, h)
    return backbone

def get_image_embeddings(entry_id:str, df:pd.DataFrame) -> np.ndarray:
    """
    データから画像特徴量を抽出する

    Parameters
    ----------
    df:pd.DataFrame
    image_size:int
    entry_id:str 画像エンコーダーの重み(cf. shopee/registry.py)

    Returns
    -------
    embeddings: np.ndarray 画像特徴量
    """
    entry = shopee.registry.get_entry_by_id(entry_id)
    if entry.model_type == "keras-origin":
        print(entry.to_dict())
        model = entry.classname(
            include_top = False,
            pooling     = "avg",
            weights     = entry.weights_filepath
        )
    elif entry.id in ["effnet_b3_swav_weights_keras"]:
        model = get_efficientnet_backbone()
        model.load_weights(entry.weighs_filepath)
    else:
        raise ValueError(f"Undefined weights name: {entry_id}")
    embeddings = model.predict(
        predict_generator(
            df                  = df, 
            image_size          = entry.image_size,
            preprocess_input    = entry.preprocess_input
        ),
        verbose=True
    )
    return embeddings

