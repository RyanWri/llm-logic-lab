from typing import Tuple
import pandas as pd
import numpy as np


def extract_title(df: pd.DataFrame, col: str) -> pd.Series:
    titles = df[col].str.extract(" ([A-Za-z]+)\.", expand=False)
    print(pd.crosstab(titles, df["Sex"]))
    return titles


def classify_title(df: pd.DataFrame, col: str) -> pd.Series:
    df[col] = df[col].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    df[col] = df[col].replace(["Ms", "Mlle"], "Miss")
    df[col] = df[col].replace("Mme", "Mrs")
    return df[col].values


def categorial_to_ordinal(df: pd.DataFrame, col: str, mapping: dict) -> pd.Series:
    df[col] = df[col].map(mapping)
    df[col] = df[col].fillna(0)
    return df[col].values


def guess_age_from_distribution(df: pd.DataFrame) -> pd.Series:
    guess_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[(df["Sex"] == i) & (df["Pclass"] == j + 1)]["Age"].dropna()
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    print(guess_ages)
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1), "Age"] = (
                guess_ages[i, j]
            )
    return df["Age"].values


def split_age_to_ranges(df: pd.DataFrame, bins: int) -> pd.Series:
    df["AgeBand"] = pd.cut(df["Age"], bins=bins)
    print(
        df[["AgeBand", "Survived"]]
        .groupby(["AgeBand"], as_index=False)
        .mean()
        .sort_values(by="AgeBand", ascending=True)
    )
    df.loc[df["Age"] <= 16, "Age"] = 0
    df.loc[(df["Age"] > 16) & (df["Age"] <= 32), "Age"] = 1
    df.loc[(df["Age"] > 32) & (df["Age"] <= 48), "Age"] = 2
    df.loc[(df["Age"] > 48) & (df["Age"] <= 64), "Age"] = 3
    df.loc[df["Age"] > 64, "Age"] = 4
    return df["AgeBand"]


def calc_family_size(df: pd.DataFrame) -> pd.Series:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print(
        df[["FamilySize", "Survived"]]
        .groupby(["FamilySize"], as_index=False)
        .mean()
        .sort_values(by="Survived", ascending=False)
    )

    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1
    print(df[["IsAlone", "Survived"]].groupby(["IsAlone"], as_index=False).mean())
    return df["FamilySize"]

def port_to_number(df: pd.DataFrame, col: str) -> pd.Series:
    ports_mapping = {"S": 0, "C": 1, "Q": 2}
    freq_port = df[col].dropna().mode()[0]
    print(f"frequent port is {freq_port}")
    df[col] = df[col].fillna(freq_port)
    print(df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    df[col] = df[col].map(ports_mapping)
    return df[col].values

def split_fare_price_to_ranges(df: pd.DataFrame, bins: int) -> pd.Series:
    df["FareBand"] = pd.qcut(df["Fare"], bins)
    print(
        df[["FareBand", "Survived"]]
        .groupby(["FareBand"], as_index=False)
        .mean()
        .sort_values(by="FareBand", ascending=True)
    )
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    return df["FareBand"]