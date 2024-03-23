from pathlib import Path
import os
from typing import List
import pandas as pd
from plot_utils import plot_histogram


def print_dataframe_stats(df: pd.DataFrame, dataset_type: str) -> None:
    print(f"Stats for {dataset_type}:")
    print(f"columns are: {df.columns.values}")
    print(df.describe())
    print(df.info())
    print(df.isnull().sum())
    print(df.describe(include=["O"]))


def correlation_to_target(
    df: pd.DataFrame, src_col: str, target_col: str
) -> pd.DataFrame:
    return train_df[[src_col, target_col]].groupby(
        [src_col], as_index=False
    ).mean().sort_values(by=target_col, ascending=False)


def clean_nan_values(df):
    df = df.dropna()
    return df


def clean_outliers(df):
    df = df[(df["Age"] > 0) & (df["Age"] < 100)]
    return df


def remove_irrelevant_features(
    df: pd.DataFrame, cols_to_remove: List[str]
) -> pd.DataFrame:
    return df.drop([cols_to_remove], axis=1)


def read_data(filepath: str, dataset_type: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(filepath, "data", f"{dataset_type}.csv"))


if __name__ == "__main__":
    filepath = Path(os.path.abspath(__file__)).parent.parent
    train_df = read_data(filepath, dataset_type="train")
    # print_dataframe_stats(train_df, dataset_type="train")
    target_col = "Survived"
    for col in ["Pclass", "Sex", "SibSp", "Parch"]:
        print(correlation_to_target(train_df, src_col=col, target_col=target_col))
    plot_histogram(train_df, col_name="Age", target_col=target_col, bins=20)