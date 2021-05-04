import numpy as np
import scipy as sp
from typing import Union, Tuple

def get_neighbors(
    embeddings:Union[np.ndarray, sp.sparse.csr.csr_matrix],
    max_candidates:int,
    chunk_size:int=100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    各要素の近傍となる候補リストを取得する
    """
    n = embeddings.shape[0]
    max_candidates = min(n, max_candidates)
    indices = np.zeros((n, max_candidates), dtype=np.int32)
    distances = np.zeros(indices.shape, dtype=np.float32)
    for i in range(0, n, chunk_size):
        similarity = embeddings[i:i+chunk_size] @ embeddings.T
        if type(similarity) == sp.sparse.csr.csr_matrix:
            similarity = similarity.toarray()
        indices[i:i+chunk_size] = np.argsort(-similarity, axis=1)[:,:max_candidates]
        distances[i:i+chunk_size] = 1 + np.sort(-similarity, axis=1)[:,:max_candidates]
    return indices, distances


