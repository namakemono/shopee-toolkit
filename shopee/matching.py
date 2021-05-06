import os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

def get_tfidf(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame
):
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    tfidf = TfidfVectorizer().fit(df["title"].str.lower())
    text_embeddings = shopee.text_embeddings.get_text_embeddings(df, tfidf)
    return text_embeddings

def get_effnet_b3(
    train_df:pd.DataFrame,      # 訓練データ
    test_df:pd.DataFrame,       # テストデータ
    image_size:int,             # 画像サイズ(256, 512)
    weights_name:str,           # 重みの名称(cf. shopee.registry)
    normed:bool=True,           # 正規化するかどうか
) -> np.ndarray:
    train_embeddings = shopee.image_embeddings.get_image_embeddings(
        train_df,
        image_size      = image_size,
        weights_name    = weights_name
    )
    train_embeddings = train_embeddings[train_df.index]
    test_embeddings = shopee.image_embeddings.get_image_embeddings(
        test_df,
        image_size      = image_size,
        weights_name    = weights_name
    )
    embeddings = np.concatenate([
        train_embeddings,
        test_embeddings
    ])
    if normed:
        embeddings = shopee.normalization.normalize(embeddings)
    return embeddings

def make_candidates(
    config,
    train_df:pd.DataFrame,
    test_df:pd.DataFrame,
    use_cache:bool,
    encoder_names=["effnet_b3", "tfidf"]
 ):
    max_candidates = config.max_candidates
    train_image_embeddings_filepath = config.train_image_embeddings_filepath
    image_size = config.image_size
    weights_name = config.weights_name

    # embeddingsの算出
    embeddings_list = []
    for encoder_name in encoder_names:
        print(f"Calculate embeddings with {encoder_name}")
        if encoder_name == "tfidf":
            embeddings = get_tfidf(train_df, test_df)
        elif encoder_name == "effnet_b3":
            embeddings = get_effnet_b3(train_df, test_df, image_size, weights_name)
        else:
            raise ValueError(f"Undefined encoder name: {encoder_name}")
        embeddings_list.append(embeddings)
    
    # 候補点の近傍となる要素を抽出
    indices_list = []
    for embeddings in embeddings_list:
        indices = get_neighbors(
            embeddings=embeddings,
            max_candidates=max_candidates
        )
        indices_list.append(indices)
   
    # マージして候補となるインデックスを
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
                record[f"feat_{encoder_names[k]}"] = distances[i][j]
            records.append(record)
    pair_df = pd.DataFrame(records)
    return pair_df


