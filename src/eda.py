from pathlib import Path
import os
from typing import List
import pandas as pd
from plot_utils import *
from engineer_utils import *


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


def read_data(filepath: str, dataset_type: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(filepath, "data", f"{dataset_type}.csv"))

def plot_eda(df: pd.DataFrame, target_col: str) -> None:
    plot_histogram(df, src_col="Age", target_col=target_col, bins=20)
    plot_histogram_with_legend(df, src_col="Pclass", target_col=target_col)
    plot_pointplot(df, src_col="Embarked", target_col=target_col)
    plot_barplot(df, src_col="Embarked", target_col=target_col)


if __name__ == "__main__":
    # read data
    filepath = Path(os.path.abspath(__file__)).parent.parent
    train_df = read_data(filepath, dataset_type="train")
    
    # show features and correlation to survived target label
    target_col = "Survived"
    # print_dataframe_stats(train_df, dataset_type="train")
    # for col in ["Pclass", "Sex", "SibSp", "Parch"]:
    #     print(correlation_to_target(train_df, src_col=col, target_col=target_col))
    

    # exploratory data analysis plots
    # plot_eda(df=train_df, target_col="Survived")

    # removing unused features
    print("Before", train_df.shape)
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    print("After", train_df.shape)

    # feature engineering
    train_df["Title"] = extract_title(train_df, col="Name")
    print(train_df.head())