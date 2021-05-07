from typing import Union
import numpy as np
import scipy as sp
import scipy.sparse
import torch
from tqdm import tqdm

# cf. https://www.kaggle.com/nicksergievskiy/pytorch-is-all-you-need-tfidf
# cf. https://www.kaggle.com/kami634/n-submission-pipline-train-test?scriptVersionId=61786179
def get_neighbors_pytorch(
    embeddings:Union[np.ndarray, sp.sparse.csr.csr_matrix],
    max_candidates:int,
    chunk_size:int=100
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(embeddings) != np.ndarray:
        embeddings = embeddings.toarray()
    embeddings = embeddings.astype(np.float16)
    embeddings=torch.from_numpy(embeddings).to(device) #.half()
    n = embeddings.shape[0]
    max_candidates = min(n, max_candidates)
    indices = np.zeros((n, max_candidates), dtype=np.int32)
    for i in tqdm(range(0, n, chunk_size)):
        similarity = torch.matmul(embeddings[i:i+chunk_size], embeddings.T)
        indices[i:i+chunk_size] = torch.argsort(-similarity, axis=1)[:,:max_candidates].cpu().numpy()
    return indices

def get_neighbors(
    embeddings:Union[np.ndarray, sp.sparse.csr.csr_matrix],
    max_candidates:int,
    chunk_size:int=100,
    use_fast:bool=False
):
    if use_fast:
        return get_neighbors_pytorch(
            embeddings,
            max_candidates,
            chunk_size
        )
    else:
        n = embeddings.shape[0]
        max_candidates = min(n, max_candidates)
        indices = np.zeros((n, max_candidates), dtype=np.int32)
        for i in range(0, n, chunk_size):
            similarity = embeddings[i:i+chunk_size] @ embeddings.T
            if type(similarity) != np.ndarray:
                similarity = similarity.toarray()
            indices[i:i+chunk_size] = np.argsort(-similarity, axis=1)[:,:max_candidates]
        return indices


