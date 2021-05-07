import re
import unidecode
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_text_embeddings(df:pd.DataFrame, tfidf:TfidfVectorizer) -> np.ndarray:
    df["title"] = df["title"].apply(clean_text)
    embeddings = tfidf.transform(df["title"].str.lower())
    return embeddings


