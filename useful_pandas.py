import numpy as np
import pandas as pd
from io import StringIO


def describe(df: pd.DataFrame, numeric_only: bool = False) -> pd.DataFrame:
    if not numeric_only:
        return (
            df.describe(include="all")
            .round(2)
            .T.fillna(df.agg("nunique").T.to_frame("unique"))
            .assign(unique=lambda df: df["unique"].astype(int))
            .join(df.isnull().sum().to_frame("missing"))
            .join(df.dtypes.to_frame("type"))
            .join(
                (df.memory_usage(deep=True) / (1024 * 1024))
                .to_frame("mem_usage")
                .round(3)
                .astype(str)
                + " MB"
            )
            .fillna("-")
        )
    else:
        return (
            df.describe(include=None)
            .round(2)
            .T.join(df.isnull().sum().to_frame("missing"))
            .join(df.dtypes.to_frame("type"))
            .join(
                (df.memory_usage(deep=True) / (1024 * 1024))
                .to_frame("mem_usage")
                .round(3)
                .astype(str)
                + " MB"
            )
        )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def plot_missing(df: pd.DataFrame, numeric_only: bool = False) -> None:
    colours = [
        "#000099",
        "#ffff00",
    ]  # specify the colours - yellow is missing. blue is not missing.
    sns.heatmap(df[cols].isnull().T, cmap=sns.color_palette(colours))


def drop_missing(df, thresh=0.8):
    df = df.dropna(axis=1, thresh=thresh * len(df))
    return df


def remove_outliers(df, column_name, lbound=0.05, ubound=0.95):
    lo = np.quantile(df[column_name], lbound)
    hi = np.quantile(df[column_name], ubound)
    df = df[df[column_name].between(lo, hi, inclusive="both")]
    return df


def series_to_category(series, unique_thresh=0.05):
    ratio = series.value_counts().shape[0] / series.shape[0]
    return series.astype("category") if ratio < unique_thresh else series


def reduce_mem_usage(df, unique_thresh=0.05):
    df = df.copy(deep=True)
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type.name not in ("object", "category"):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max
                ):
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max
                ):
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif (
                    c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max
                ):
                    df[col] = df[col].astype(np.uint64)
            elif str(col_type)[:5] == "float":
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type.name == "object":
            df[col] = series_to_category(df[col], unique_thresh=unique_thresh)

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def read_csv(csv: str) -> pd.DataFrame:
    """
    Read a CSV string into a DataFrame (useful where pd.read_clipboard isn't available (google colab / non-local notebooks)).
    """
    return pd.read_csv(StringIO(csv))


def merge_csvs(csv_list: list) -> pd.DataFrame:
    """
    Merge a list of CSV files into a single DataFrame.
    To get a list use something like: # from glob import glob; csv_list = sorted(glob('*.csv'))
    """
    return pd.concat(
        [pd.read_csv(file).assign(filename=file) for file in files], ignore_index=True
    )  # map(pd.read_parquet, files) if the source file isn't needed


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/gchoi/Dataset/master/DirectMarketing.csv"

    marketing_cleaned = (
        pd.read_csv(url)
        .infer_objects()
        .pipe(clean_columns)
        .pipe(drop_missing)
        .pipe(remove_outliers, "salary")
    )

    print(marketing_cleaned.head(5), "\n")
    print(describe(marketing_cleaned))
    print(describe(marketing_cleaned, numeric_only=True))
