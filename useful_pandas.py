import numpy as np
import pandas as pd
import seaborn as sns
from io import StringIO
from sqlalchemy.engine.base import Engine


def describe(df: pd.DataFrame, include: str='all', percentiles: list=[.25, .5, .75], rounding: int=2) -> pd.DataFrame:
    def keep_relevant_cols(df):
        cols = ['mem_usage', 'dtype', 'count', 'unique', 'missing', 'top', 'freq', 'mean', 'std', 'min']
        cols = cols + [f'{p:.0%}' for p in percentiles] + ['max']
        cols = [c for c in cols if c in df.columns]
        return df[cols]

    def fmt_numbers(df):
        map_ints = {'count': 'int64', 'missing': 'int64', 'unique': 'int64'}
        map_ints = {k: v for k, v in map_ints.items() if k in df.columns}
        return df.round(rounding).astype(map_ints)

    try:
        mem_usage = (df.memory_usage(deep=True) / (1024 * 1024)).to_frame('mem_usage').round(rounding).astype(str) + ' MB'
        dtype = df.dtypes.to_frame('dtype')
        missing = df.isnull().sum().to_frame('missing')
        add_extra = lambda df: df.T.join(mem_usage).join(dtype).join(missing)
        
        output = df.describe(include=include, percentiles=percentiles, datetime_is_numeric=True).pipe(add_extra)

        if 'unique' in output.columns:
            fill_unique = df.agg('nunique').T.to_frame('unique')
            output = output.fillna(fill_unique)
        
        return output.pipe(keep_relevant_cols).pipe(fmt_numbers).fillna('-')

    except ValueError as e:
        if e.args[0] != 'No objects to concatenate':
            raise e
        datatypes = {str(d) for d in df.dtypes.to_list()}
        print(f"The DataFrame has no '{include}' columns, only: {datatypes}.\n\nShowing an include='all' summary instead:\n")
        return describe(df, include='all')


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def plot_missing(df: pd.DataFrame, numeric_only: bool = False) -> None:
    colours = ["#000099", "#ffff00"] # yellow missing. blue present
    sns.heatmap(df.isnull().T, cmap=sns.color_palette(colours))


def drop_missing(df, thresh=0.8):
    return df.dropna(axis=1, thresh=thresh * len(df))


def remove_outliers(df, column_name, lbound=0.05, ubound=0.95):
    lo = np.quantile(df[column_name], lbound)
    hi = np.quantile(df[column_name], ubound)
    df = df[df[column_name].between(lo, hi, inclusive="both")]
    return df


def downcast(df: pd.DataFrame, unique_thresh: float = 0.05) -> pd.DataFrame:
    """Compress common dtypes to save memory (RAM)."""
    mem_before = df.memory_usage(deep=True).sum()
    mem_before_mb = round(mem_before / (1024**2), 2)
    df = df.convert_dtypes()

    for column in df.select_dtypes(["string", "object"]):
        if (len(df[column].unique()) / len(df[column])) < unique_thresh:
            df[column] = df[column].astype("category")

    for column in df.select_dtypes(["float"]):
        df[column] = pd.to_numeric(df[column], downcast="float")

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


def read_csv(csv: str, **kwargs) -> pd.DataFrame:
    """Read CSV string to DataFrame, useful when pd.read_clipboard isn't available (e.g. google-colab/databricks)"""
    return pd.read_csv(StringIO(csv), **kwargs)


def concat_csvs(files: list, keep_filename: bool=False, ignore_index: bool=True, **kwargs) -> pd.DataFrame:
    """Read and concat a list of CSVs (from filepaths) to a single DataFrame. Get a list using: # from glob import glob; csvs = sorted(glob('*.csv'))"""
    if keep_filename:
        dfs = [pd.read_csv(file).assign(filename=file) for file in files]
        return pd.concat(dfs, ignore_index=ignore_index, **kwargs)
    return pd.concat(map(pd.read_csv, files), ignore_index=ignore_index, **kwargs)


def fast_insert(df: pd.DataFrame, table_name: str, engine: Engine, chunksize: int=100_000, if_exists: str="replace") -> bool:
    # create or replace table with schema of the DataFrame
    df.head(0).to_sql(table_name, engine, if_exists=if_exists, index=False)

    try:
        conn = engine.raw_connection() # direct, low-level connection to the database
        with conn.cursor() as cur:
            # use StringIO to write DataFrame as in-memory CSV
            output = StringIO()

            # chunk through DataFrame and write to PostgreSQL
            for (idx, chunk) in df.groupby(np.arange(len(df)) // chunksize):
                # write chunk to StringIO object
                chunk.to_csv(output, sep='\t', header=False, index=False)
                output.seek(0)

                # use COPY command to load into PostgreSQL
                cur.copy_from(output, table_name, null='')
                conn.commit()

                # reset StringIO object for the next chunk
                output.seek(0)
                output.truncate(0)
    except Exception as e:
        return False # failure
    finally:
        conn.close()
        return True # success


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/gchoi/Dataset/master/DirectMarketing.csv"

    marketing_cleaned = (
        pd.read_csv(url)
        .infer_objects()
        .pipe(clean_columns)
        .pipe(drop_missing)
        .pipe(remove_outliers, "salary")
    )

    print(marketing_cleaned.head(5))
    print(describe(marketing_cleaned, include=None))
    print(describe(marketing_cleaned, include='all'))
