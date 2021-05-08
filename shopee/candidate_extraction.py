from typing import Union
import numpy as np
import scipy as sp
import scipy.sparse
from tqdm import tqdm
import scipy.sparse
import torch
from scipy.sparse import coo_matrix, hstack, csr_matrix

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

def convert_sp_to_torch(embeddings:csr_matrix) -> torch.sparse.FloatTensor:
    coo = embeddings.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    embeddings = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return embeddings

def get_neighbors_pytorch_sparse(
    embeddings:Union[np.ndarray, sp.sparse.csr.csr_matrix],
    max_candidates:int,
    chunk_size:int=100
):
    print(type(embeddings))
    if type(embeddings) == sp.sparse.csr.csr_matrix: # coo とかだと現在できないので注意
        print("ok")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n = embeddings.shape[0]
        max_candidates = min(n, max_candidates)
        indices = np.zeros((n, max_candidates), dtype=np.int32)
        embeddings_torch = convert_sp_to_torch(embeddings).to(device)
        with torch.no_grad():
            for i in tqdm(range(0, n, chunk_size)):
                embeddings_torchT = torch.from_numpy(
                    embeddings[i:i+chunk_size].toarray().T.astype(np.float32)
                ).to(device) 
                similarity = torch.sparse.mm(embeddings_torch, embeddings_torchT)
                indices[i:i+chunk_size] = torch.argsort(
                    -similarity, axis=0
                ).cpu().numpy().T[:,:max_candidates]
        return indices
    else:
        print("no")
        return get_neighbors_pytorch(
            embeddings,
            max_candidates,
            chunk_size
        )

def get_neighbors(
    embeddings:Union[np.ndarray, sp.sparse.csr.csr_matrix],
    max_candidates:int,
    chunk_size:int=100,
    use_fast:bool=False
):
    if use_fast:
        return get_neighbors_pytorch_sparse(
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


