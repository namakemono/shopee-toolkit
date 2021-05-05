import numpy as np
import pandas as pd

def f1_score(y_true:pd.Series, y_pred:pd.Series):
    """
    各行のF1スコアを計算する

    examples
    --------
    >>> metrics.f1_score(pd.Series(["A B C", "D E"]), pd.Series(["B D", "D E"]))
    array([0.4, 1. ])
    """
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

def calc_f1_score(df:pd.DataFrame, pair_df:pd.DataFrame) -> float:
    _df = pair_df[pair_df["prediction"] == 1]
    if len(_df) == 0:
        return 0
    gdf = _df.groupby(
        "posting_id",
        as_index=False
    )["candidate_posting_id"].apply(
        lambda _: " ".join(_)
    ).rename(columns={
        "candidate_posting_id": "predictions"
    })
    s = df.groupby("label_group")["posting_id"].apply(lambda _: " ".join(_))
    df["matches"] = df["label_group"].map(s)
    # F1
    _df = pd.merge(
        df[["posting_id", "matches"]],
        gdf[["posting_id", "predictions"]],
        on="posting_id",
        how="left"
    )
    _df["score"] = f1_score(_df["matches"], _df["predictions"].fillna(""))
    return _df["score"].mean()


def show_score(df:pd.DataFrame, pair_df:pd.DataFrame):
    y_true = pair_df["matched"]
    y_pred = pair_df["prediction"]

    total = df.groupby("label_group")["posting_id"].apply(lambda _: len(_)**2).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = total - tp
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    f1 = calc_f1_score(df, pair_df)

    print("positive ratio: %.4f" % y_true.mean())
    print("total: %d" % total)
    print("TP: %d" % tp)
    print("FP: %d" % fp)
    print("FN: %d" % fn)
    print("TN: %d" % tn)
    print("F1: %.4f" % f1)



