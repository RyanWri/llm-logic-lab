# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


filepath = Path(os.path.abspath(__file__)).parent.parent
directory = os.path.join(filepath, "eda_plots")


def save_plot(plt: plt, filepath: str, plot_name: str) -> None:
    image = os.path.join(filepath, plot_name)
    plt.savefig(image)


def plot_histogram(df: pd.DataFrame, src_col: str, target_col: str, bins: int) -> None:
    g = sns.FacetGrid(df, col=target_col)
    g.map(plt.hist, src_col, bins=bins)
    save_plot(plt, directory, f"histogram_{src_col}_{target_col}.png")


def plot_histogram_with_legend(df: pd.DataFrame, src_col: str, target_col: str) -> None:
    grid = sns.FacetGrid(df, col=target_col, row=src_col)
    grid.map(plt.hist, "Age", alpha=0.5, bins=20)
    grid.add_legend()
    save_plot(plt, directory, f"histogram_{src_col}_{target_col}_legend.png")


def plot_pointplot(df: pd.DataFrame, src_col: str, target_col: str) -> None:
    grid = sns.FacetGrid(df, row=src_col)
    grid.map(sns.pointplot, "Pclass", target_col, "Sex", palette="deep")
    grid.add_legend()
    save_plot(plt, directory, f"pointplot_{src_col}_{target_col}.png")


def plot_barplot(df: pd.DataFrame, src_col: str, target_col: str) -> None:
    grid = sns.FacetGrid(df, row=src_col, col=target_col)
    grid.map(sns.barplot, "Sex", "Fare", alpha=0.5, errorbar=None)
    grid.add_legend()
    save_plot(plt, directory, f"barplot_{src_col}_{target_col}.png")


def plot_model_scores(df: pd.DataFrame) -> None:
    plt.figure(figsize=(20, 10))
    fig = plt.bar(df.index, df["Score"], color="aqua")
    plt.grid()
    save_plot(plt, directory, f"models_accuracy_scores.png")
