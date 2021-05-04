import pandas as pd

def to_common(row):
    S = set(row["matches"].split())
    T = set(row["predictions"].split())
    return " ".join(S & T)

def to_union(row):
    S = set(row["matches"].split())
    T = set(row["predictions"].split())
    return " ".join(S | T)

def merge(a, b):
    S = set(list(a))
    T = set(list(b))
    return list(S | T)

def to_posting_ids(
    df:pd.DataFrame,
    indices_list:pd.Series
) -> pd.Series:
    to_posting_id = dict(enumerate(df["posting_id"]))
    res = []
    for indices in indices_list:
        res.append(" ".join(
            [to_posting_id[index] for index in indices]
        ))
    return pd.Series(res)

