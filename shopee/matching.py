import numpy as np
import scipy as sp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import shopee

def load_image_embeddings(filepath:str) -> np.ndarray:
    return np.load(filepath)

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

def make_candidates(config, train_df:pd.DataFrame, test_df:pd.DataFrame, use_cache:bool):
    max_candidates = config.max_candidates
    train_image_embeddings_filepath = config.train_image_embeddings_filepath
    image_size = config.image_size
    weights_name = config.weights_name

    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    
    # テキストのembeddingsと候補抽出
    tfidf = TfidfVectorizer().fit(df["title"].str.lower())
    text_embeddings = shopee.text_embeddings.get_text_embeddings(df, tfidf)
    print("text embeddings", text_embeddings.shape)
    text_indices = get_neighbors(
        embeddings=text_embeddings,
        max_candidates=max_candidates
    )
    print("text indices", text_indices.shape)
    
    # 画像のembeddingsと候補抽出
    if use_cache:
        train_image_embeddings = load_image_embeddings(
            filepath=train_image_embeddings_filepath
        )
    else:
        train_image_embeddings = shopee.image_embeddings.get_image_embeddings(
            train_df,
            image_size      = image_size,
            weights_name    = weights_name
        )
    train_image_embeddings = train_image_embeddings[train_df.index]
    test_image_embeddings = shopee.image_embeddings.get_image_embeddings(
        test_df,
        image_size=image_size,
        weights_name=weights_name
    )
    image_embeddings = np.concatenate([
        train_image_embeddings,
        test_image_embeddings
    ])
    image_embeddings = shopee.normalization.normalize(image_embeddings)
    print("image embeddings", image_embeddings.shape)
    image_indices = get_neighbors(
        embeddings=image_embeddings,
        max_candidates=max_candidates
    )
    print("image indices", image_indices.shape)
    
    n = len(df)
    candidate_indices = [shopee.utils.merge(image_indices[i], text_indices[i]) for i in range(n)]
    print("candidate indices", len(candidate_indices))

    image_distances = to_distances(candidate_indices, image_embeddings)
    print("image distances", len(image_distances))
    text_distances = to_distances(candidate_indices, text_embeddings)
    print("text distances", len(text_distances))
    
    to_posting_id = dict(enumerate(df["posting_id"]))
    to_index = {v: k for k, v in to_posting_id.items()}
    s = df.groupby("label_group")["posting_id"].unique().to_dict()
    df["matches"] = df["label_group"].map(s).apply(lambda _: " ".join(_))
    df["match_indices"] = df["matches"].apply(lambda _: [to_index[k] for k in _.split()])
    match_indices = df["match_indices"].tolist()

    records = []
    for i in range(n):
        for j, img_feat, text_feat in zip(candidate_indices[i], image_distances[i], text_distances[i]):
            matched = int(j in match_indices[i])
            records.append({
                "posting_id": to_posting_id[i],
                "candidate_posting_id": to_posting_id[j],
                "img_feat": img_feat,
                "text_feat": text_feat,
                "matched": matched,
            })
    pair_df = pd.DataFrame(records)
    return pair_df


