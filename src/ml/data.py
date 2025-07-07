import pandas as pd

from typing import List
from tqdm import tqdm


def _melt(_pv, id_vars="ts", value_name="value"):
    return _pv.reset_index().melt(id_vars=id_vars, value_name=value_name)


def generate_features(
    df: pd.DataFrame,
    index_col: str = "ts",
    pivot_col: str = "ticker",
    value_col: str = "vwap",
    means: List[int] = [],
    stds: List[int] = [],
    mns: List[int] = [],
    mxs: List[int] = [],
    lags: List[int] = [],
    futs: List[int] = [],
    verbose: bool = True,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    df : pd.DataFrame
        The input pd.DataFrame

    index_col : str
        The main index that will be kept after the pivot (e.g. "ts").

    pivot_col : str
        The column used to generate new columns in the pivoted table (e.g. if "ticker",
        all unique tickers will get a column in the pivoted table).

    value_col : str
        The column that will be used as the "value" for each "ts" row and "ticker"
        column.

    means : List[int]
    stds : List[int]
    mns : List[int]
    mxs : List[int]
    lags : List[int]
    futs : List[int]

        Lists containing integers for the means, standard devaitions, lags, and future
        values to be computed.

        Today is a lag of 0. Yesterday is a lag of 1, and etc. Thus, lag cannot be less
        than 1.

        Similarly, tomorrow is a future value of 1, and two days from now is a future
        value of 2, and so future cannot be less than 1 either.

    Returns
    -------
    pd.DataFrame
        A featured pd.DataFrame containing the means, stds, lags, and futs.
    """
    r = df.copy()
    pv = r.pivot(index=index_col, columns=pivot_col, values=value_col)
    pv.sort_index(inplace=True)

    fcns = [
        lambda x: x.mean(),
        lambda x: x.std(),
        lambda x: x.min(),
        lambda x: x.max(),
    ]

    for f, l, label in tqdm(
        zip(
            fcns,
            [means, stds, mns, mxs],
            ["mean", "std", "min", "max"],
        ),
        disable=not verbose,
    ):
        for v in l:
            rolling = pv.rolling(v)
            r = r.merge(_melt(f(rolling), value_name=f"{label}_{v}"))

    for l in tqdm(lags, disable=not verbose):
        if l < 1:
            raise ValueError(
                "Values in `lags` must not be less than 1. See documentation."
            )

        r = r.merge(_melt(pv.shift(l), value_name=f"lag_{l}"))

    for f in tqdm(futs, disable=not verbose):
        if f < 1:
            raise ValueError(
                "Values in `futs` must not be less than 1. See documentation."
            )

        r = r.merge(_melt(pv.shift(-f), value_name=f"fut_{f}"))

    return r
