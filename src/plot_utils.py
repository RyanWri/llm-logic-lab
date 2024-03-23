# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_histogram(df: pd.DataFrame, col_name: str, target_col: str, bins: int) -> None:
    g = sns.FacetGrid(df, col=target_col)
    g.map(plt.hist, col_name, bins=bins)
    plt.show()