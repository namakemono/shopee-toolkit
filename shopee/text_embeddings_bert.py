import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

def build_model(
    pretrained_model_directory:str,
    max_seq_length:int
):
    input_shape = (max_seq_length, )
    x_input_ids = tf.keras.Input(shape=input_shape, dtype=tf.int32, name="input_id")
    x_attention_mask = tf.keras.Input(shape=input_shape, dtype=tf.int32, name="attention_mask")
    x_token_type_ids = tf.keras.Input(shape=input_shape, dtype=tf.int32, name="segment_id")
    bert_model = transformers.TFAutoModel.from_pretrained(
        pretrained_model_directory
    )
    outputs = bert_model(
        input_ids=x_input_ids,
        attention_mask=x_attention_mask,
        token_type_ids=x_token_type_ids
    )
    model = tf.keras.Model(
        inputs=[x_input_ids, x_attention_mask, x_token_type_ids],
        outputs=[outputs.pooler_output]
    )
    return model

def get_text_embeddings(
    df:pd.DataFrame,
    pretrained_model_directory:str,
    max_seq_length:int
):
    """
    Examples
    --------
    embeddings = get_text_embeddings(
        df=train_df,
        pretrained_model_directory="bert-base-uncased",
        max_seq_length=100
    )
    """
    # tokenization
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_directory
    )
    tokens = df["title"].str.lower().apply(lambda text: tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length"
    ))
    # tensorflow用のinput形式に変換
    shape = (len(tokens), max_seq_length)
    input_ids = np.zeros(shape, dtype=np.int32)
    attention_mask = np.zeros(shape, dtype=np.int32)
    token_type_ids = np.zeros(shape, dtype=np.int32)
    for i, token in enumerate(tokens):
        input_ids[i] = token["input_ids"]
        attention_mask[i] = token["attention_mask"]
        if "token_type_ids" in token:
            token_type_ids[i] = token["token_type_ids"]
    # embeddingsの計算
    model = build_model(
        pretrained_model_directory=pretrained_model_directory,
        max_seq_length=max_seq_length
    )
    embeddings = model.predict([input_ids, attention_mask, token_type_ids])
    return embeddings

