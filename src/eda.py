from pathlib import Path
import os
from typing import List
import pandas as pd
from plot_utils import *
from engineer_utils import *
import numpy as np
import logging
from train import Trainer
from preprocess import preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    return (
        train_df[[src_col, target_col]]
        .groupby([src_col], as_index=False)
        .mean()
        .sort_values(by=target_col, ascending=False)
    )


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
    # x = preprocess_data(df=train_df, target_col=target_col)
    # print_dataframe_stats(train_df, dataset_type="train")
    # for col in ["Pclass", "Sex", "SibSp", "Parch"]:
    #     print(correlation_to_target(train_df, src_col=col, target_col=target_col))

    # exploratory data analysis plots
    # plot_eda(df=train_df, target_col="Survived")

    # removing unused features
    print("Before", train_df.shape)
    train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
    print("After", train_df.shape)

    # feature engineering
    train_df["Title"] = extract_title(train_df, col="Name")
    train_df["Title"] = classify_title(train_df, "Title")
    # show each title to survive rate
    print(train_df[["Title", target_col]].groupby(["Title"], as_index=False).mean())

    # map categories to numbers
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    train_df["Title"] = categorial_to_ordinal(train_df, "Title", title_mapping)
    print(train_df.head())

    # now we can safely remove name and passengerID
    train_df = train_df.drop(["Name", "PassengerId"], axis=1)
    print("After removing name and passengerId", train_df.shape)

    # map gender to int
    gender_mapping = {"female": 1, "male": 0}
    train_df["Sex"] = categorial_to_ordinal(train_df, "Sex", gender_mapping).astype(int)
    print(train_df.head())

    # plot age distribution after feature engineering - Missing for now

    logger.info("guess age from distribution")
    train_df["Age"] = guess_age_from_distribution(train_df).astype(int)
    print(train_df.head())

    logger.info("split age to ranges")
    train_df["AgeBand"] = split_age_to_ranges(train_df, bins=5)
    print(train_df.head())

    logger.info("removing age band after finding ranges")
    train_df = train_df.drop(["AgeBand"], axis=1)

    logger.info("family size feature")
    train_df["FamilySize"] = calc_family_size(train_df)
    print(train_df.head())

    logger.info("remove family features")
    train_df = train_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)
    print(train_df.head())

    logger.info("age times class feature")
    train_df["Age*Class"] = train_df.Age * train_df.Pclass

    logger.info("ports feature engineering")
    train_df["Embarked"] = port_to_number(train_df, "Embarked").astype(int)
    print(train_df.head())

    logger.info("filling missing fare with median data")
    train_df["Fare"] = train_df["Fare"].fillna(train_df["Fare"].dropna().median())
    train_df["Fare"] = split_fare_price_to_ranges(train_df, bins=4)
    # remove Fare band as we have fare ranges
    train_df = train_df.drop(["FareBand"], axis=1)
    print(train_df.head())

    # training
    train_df2 = read_data(filepath, dataset_type="train")
    train_df2 = preprocess_data(train_df2, target_col=target_col)
    train_df2 = train_df2.drop("PassengerId", axis=1)
    test_df = read_data(filepath, dataset_type="test")
    test_df = preprocess_data(test_df, target_col=target_col)
    
    # logistics regression
    trainer = Trainer(train_df2, test_df)
    logreg = trainer.train_lr()

    coef_df = trainer.correlation_to_lr(train_df2, logreg)
    print(coef_df.sort_values(by='Correlation', ascending=False))


    # Support Vector Machines
    svm = trainer.train_svm()