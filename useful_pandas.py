import numpy as np
import pandas as pd


def copy_df(df):
    return df.copy()


def clean_columns(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


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


def unique_column_values(df):
    return (
        df.agg("nunique")
        .to_frame()
        .rename(columns={0: "unique_values"})
        .sort_values("unique_values")
    )


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

    print(marketing_cleaned.head(10), "\n\n\n")
    print(unique_column_values(marketing_cleaned))
