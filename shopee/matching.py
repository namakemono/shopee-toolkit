import os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import shopee


def get_neighbors(embeddings, max_candidates:int, chunk_size:int=100):
    n = embeddings.shape[0]
    max_candidates = min(n, max_candidates)
    indices = np.zeros((n, max_candidates), dtype=np.int32)
    for i in range(0, n, chunk_size):
        similarity = embeddings[i:i+chunk_size] @ embeddings.T
        if type(similarity) != np.ndarray:
            similarity = similarity.toarray()
        indices[i:i+chunk_size] = np.argsort(-similarity, axis=1)[:,:max_candidates]
    return indices

def to_distances(indices, embeddings):
    n = len(indices)
    distances = []
    for i in range(n):
        distance = embeddings[i] @ embeddings[indices[i]].T
        if type(distance) != np.ndarray:
            distance = distance.toarray().flatten()
        distances.append(distance)
    return distances

def get_tfidf_embeddings(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame
):
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    tfidf = TfidfVectorizer().fit(df["title"].str.lower())
    text_embeddings = shopee.text_embeddings.get_text_embeddings(df, tfidf)
    return text_embeddings

def get_image_embeddings(
    entry_id:str,  
    train_df:pd.DataFrame,      # 訓練データ
    test_df:pd.DataFrame,       # テストデータ
    normed:bool=True,           # 正規化するかどうか
) -> np.ndarray:
    entry = shopee.registry.get_entry_by_id(entry_id)
    if os.path.exists(entry.embeddings_filepath):
        train_embeddings = shopee.image_embeddings.load_image_embeddings(
            entry.embeddings_filepath
        )
    else:
        train_embeddings = shopee.image_embeddings.get_image_embeddings(
            entry_id    = entry_id,
            df          = train_df,
        )
    test_embeddings = shopee.image_embeddings.get_image_embeddings(
        entry_id    = entry_id,
        df          = test_df,
    )
    embeddings = np.concatenate([
        train_embeddings,
        test_embeddings
    ])
    if normed:
        embeddings = shopee.normalization.normalize(embeddings)
    return embeddings

def make_candidates(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame,
    use_cache:bool,
    entry_ids:List[str],
    max_candidates:int
 ) -> pd.DataFrame:
    # embeddingsの算出
    embeddings_list = []
    for entry_id in entry_ids:
        print(f"Calculate embeddings with {entry_id}")
        if entry_id == "tfidf-v1":
            embeddings = get_tfidf_embeddings(train_df, test_df)
        else:
            embeddings = get_image_embeddings(entry_id, train_df, test_df)
        embeddings_list.append(embeddings)
    
    # 候補点の近傍となる要素を抽出
    indices_list = []
    for embeddings in embeddings_list:
        indices = get_neighbors(
            embeddings=embeddings,
            max_candidates=max_candidates
        )
        indices_list.append(indices)
   
    # マージして候補となるインデックスを作る
    candidate_indices = indices_list[0]
    for indices in indices_list:
        candidate_indices = [shopee.utils.merge(candidate_indices[i], indices[i]) for i in range(len(indices))]
    print("candidate indices", len(candidate_indices))

    # 距離を求める
    print("calculate distance")
    distances_list = []
    for embeddings in embeddings_list:
        distances_list.append(
            to_distances(candidate_indices, embeddings)
        )

    # 正解情報を付け加える
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    to_posting_id = dict(enumerate(df["posting_id"]))
    to_index = {v: k for k, v in to_posting_id.items()}
    s = df.groupby("label_group")["posting_id"].unique().to_dict()
    df["matches"] = df["label_group"].map(s).apply(lambda _: " ".join(_))
    df["match_indices"] = df["matches"].apply(lambda _: [to_index[k] for k in _.split()])
    match_indices = df["match_indices"].tolist()

    records = []
    for i in range(len(candidate_indices)):
        for j, index in enumerate(candidate_indices[i]):
            matched = int(index in match_indices[i])
            record = {
                "posting_id": to_posting_id[i],
                "candidate_posting_id": to_posting_id[index],
                "matched": matched,
            }
            for k, distances in enumerate(distances_list):
                record[f"feat_{entry_ids[k]}"] = distances[i][j]
            records.append(record)
    pair_df = pd.DataFrame(records)
    return pair_df


