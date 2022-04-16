import numpy as np
import pandas as pd
import seaborn as sns
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
    sns.heatmap(df.isnull().T, cmap=sns.color_palette(colours))


def drop_missing(df, thresh=0.8):
    df = df.dropna(axis=1, thresh=thresh * len(df))
    return df


def remove_outliers(df, column_name, lbound=0.05, ubound=0.95):
    lo = np.quantile(df[column_name], lbound)
    hi = np.quantile(df[column_name], ubound)
    df = df[df[column_name].between(lo, hi, inclusive="both")]
    return df


def downcast(df: pd.DataFrame, unique_thresh: float = 0.05) -> pd.DataFrame:
    '''Compression of the common dtypes "float64", "int64", "object" or "string"'''
    mem_before = df.memory_usage(deep=True).sum()
    mem_before_mb = round(mem_before / (1024**2), 2)

    # convert the dataframe columns to ExtensionDtype (e.g. object to string, or 1.0 float to 1 integer, etc.)
    df = df.convert_dtypes()

    # string categorization (only the ones with low cardinality)
    for column in df.select_dtypes(["string", "object"]):
        if (len(df[column].unique()) / len(df[column])) < unique_thresh:
            df[column] = df[column].astype("category")

    for column in df.select_dtypes(["float"]):
        df[column] = pd.to_numeric(df[column], downcast="float")

    # int64 downcasting (depending if negative values are apparent (='signed') or only >=0 (='unsigned'))
    for column in df.select_dtypes(["integer"]):
        if df[column].min() >= 0:
            df[column] = pd.to_numeric(df[column], downcast="unsigned")
        else:
            df[column] = pd.to_numeric(df[column], downcast="signed")

    mem_after = df.memory_usage(deep=True).sum()
    mem_after_mb = round(mem_after / (1024**2), 2)
    compression = round(((mem_before - mem_after) / mem_before) * 100)

    print(
        f"DataFrame compressed by {compression}% from {mem_before_mb} MB down to {mem_after_mb} MB."
    )

    return df


def read_csv(csv: str) -> pd.DataFrame:
    """
    Read a CSV string into a DataFrame (useful where pd.read_clipboard isn't available (google colab / non-local notebooks)).
    """
    return pd.read_csv(StringIO(csv))


def merge_csvs(files: list) -> pd.DataFrame:
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
