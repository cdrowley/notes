import numpy as np
import pandas as pd


def clean_columns(df):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def describe(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.describe()
        .T.round(1)
        .join(df.isnull().sum().to_frame("missing"))
        .join(df.agg("nunique").T.to_frame("unique"))
        .join(df.dtypes.to_frame("type"))
    )


def drop_missing(df, thresh=0.8):
    df = df.dropna(axis=1, thresh=thresh * len(df))
    return df


def remove_outliers(df, column_name, lbound=0.05, ubound=0.95):
    lo = np.quantile(df[column_name], lbound)
    hi = np.quantile(df[column_name], ubound)
    df = df[df[column_name].between(lo, hi, inclusive="both")]
    return df


def to_category(df, unique_thresh=0.05):
    cols = df.select_dtypes(include="object").columns
    for col in cols:
        ratio = len(df[col].value_counts()) / len(df)
        if ratio < unique_thresh:
            df[col] = df[col].astype("category")
    return df


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/gchoi/Dataset/master/DirectMarketing.csv"

    marketing_cleaned = (
        pd.read_csv(url)
        .infer_objects()
        .pipe(clean_columns)
        .pipe(drop_missing)
        .pipe(remove_outliers, "salary")
        .pipe(to_category)
    )

    print(marketing_cleaned.head(5), "\n")
    print(describe(marketing_cleaned))
