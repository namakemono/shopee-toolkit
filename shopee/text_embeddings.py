import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_text_embeddings(df:pd.DataFrame, tfidf:TfidfVectorizer) -> np.ndarray:
    embeddings = tfidf.transform(df["title"].str.lower())
    return embeddings

def get_neighbors(embeddings:np.ndarray, max_candidates:int):
    similarity = (x @ x.T).toarray()
    indices = np.argsort(-similarity)[:,:max_candidates]
    return indices


