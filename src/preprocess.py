import pandas as pd
from engineer_utils import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # show features and correlation to survived target label
    # print_dataframe_stats(df, dataset_type="train")
    # for col in ["Pclass", "Sex", "SibSp", "Parch"]:
    #     print(correlation_to_target(df, src_col=col, target_col=target_col))

    # exploratory data analysis plots
    # plot_eda(df=df, target_col="Survived")

    # removing unused features
    print("Before", df.shape)
    df = df.drop(["Ticket", "Cabin"], axis=1)
    print("After", df.shape)

    # feature engineering
    df["Title"] = extract_title(df, col="Name")
    df["Title"] = classify_title(df, "Title")
    # map categories to numbers
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df["Title"] = categorial_to_ordinal(df, "Title", title_mapping)

    # now we can safely remove name
    df = df.drop(["Name"], axis=1)
    print("After removing name", df.shape)

    # map gender to int
    gender_mapping = {"female": 1, "male": 0}
    df["Sex"] = categorial_to_ordinal(df, "Sex", gender_mapping).astype(int)
    print(df.head())

    # plot age distribution after feature engineering - Missing for now

    logger.info("guess age from distribution")
    df["Age"] = guess_age_from_distribution(df).astype(int)
    print(df.head())

    logger.info("split age to ranges")
    df["AgeBand"] = split_age_to_ranges(df, bins=5)
    print(df.head())

    logger.info("removing age band after finding ranges")
    df = df.drop(["AgeBand"], axis=1)

    logger.info("family size feature")
    df["FamilySize"] = calc_family_size(df)
    print(df.head())

    logger.info("remove family features")
    df = df.drop(["Parch", "SibSp", "FamilySize"], axis=1)
    print(df.head())

    logger.info("age times class feature")
    df["Age*Class"] = df.Age * df.Pclass

    logger.info("ports feature engineering")
    df["Embarked"] = port_to_number(df, "Embarked").astype(int)
    print(df.head())

    logger.info("filling missing fare with median data")
    df["Fare"] = df["Fare"].fillna(df["Fare"].dropna().median())
    df["Fare"] = split_fare_price_to_ranges(df, bins=4)
    # remove Fare band as we have fare ranges
    df = df.drop(["FareBand"], axis=1)
    print(df.head())
    return df
