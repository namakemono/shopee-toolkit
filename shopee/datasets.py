import os
import pandas as pd
from sklearn.model_selection import GroupKFold

def to_matches(df:pd.DataFrame) -> pd.Series:
    s = df.groupby("label_group")["posting_id"].unique().to_dict()
    return df["label_group"].map(s).apply(lambda _: " ".join(_))

def load_train_data() -> pd.DataFrame:
    df = pd.read_csv("../input/shopee-product-matching/train.csv")
    df["filepath"] = df["image"].apply(
        lambda _: os.path.join("../input/shopee-product-matching/train_images", _)
    )
    df["matches"] = to_matches(df)
    gkf = GroupKFold(n_splits = 10)
    df['fold'] = -1
    for kfold_index, (_, valid_index) in enumerate(gkf.split(df, groups=df["label_group"].tolist())):
        df.loc[valid_index, 'fold'] = kfold_index
    return df

def load_test_data() -> pd.DataFrame:
    df = pd.read_csv("../input/shopee-product-matching/test.csv")
    df["filepath"] = df["image"].apply(
        lambda _: os.path.join("../input/shopee-product-matching/test_images", _)
    )
    # label_groupは不明なので，暫定的にimage_phashをグループとして割り当てる
    df["label_group"] = df["image_phash"]
    return df

