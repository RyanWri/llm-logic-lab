import pandas as pd


def extract_title(df: pd.DataFrame, col: str)-> pd.Series:
    titles = df[col].str.extract(' ([A-Za-z]+)\.', expand=False)
    print(pd.crosstab(titles, df['Sex']))
    return titles