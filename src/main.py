# This is the kernel i am based upon
# https://www.kaggle.com/code/soham1024/titanic-data-science-eda-with-meme-solution
import logging

import numpy as np
import pandas as pd

from plot_utils import plot_model_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from preprocess import read_data, preprocess_data
from train import Trainer


def submit_results(test_df: pd.DataFrame, best_model_predictions: np.ndarray) -> None:
    # after reviewing all models we find random forest to be the best
    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": best_model_predictions}
    )
    submission.to_csv("submission2.csv", index=False)


def main():
    logger.info("Preprocess")
    # training
    train_df = read_data(csv_file="train.csv")
    train_df = preprocess_data(train_df)
    train_df = train_df.drop("PassengerId", axis=1)
    test_df = read_data(csv_file="test.csv")
    test_df = preprocess_data(test_df)
    logger.info("********* Training ************")
    # logistics regression
    trainer = Trainer(train_df, test_df)
    logreg = trainer.train_lr()

    coef_df = trainer.correlation_to_lr(train_df, logreg)
    print(coef_df.sort_values(by="Correlation", ascending=False))

    # Support Vector Machines
    svm = trainer.train_svm()

    # K Nearest Neighbors with 3 neighbours
    knn = trainer.train_knn(neighbors=3)

    # Gaussian Naive Bayes
    naive_bayes = trainer.train_naive_bayes()

    # Perceptron
    perceptron = trainer.train_perceptron()

    # Linear SVC
    linear_svc = trainer.train_linear_svc()

    # Stochastic Gradient Descent
    sgd = trainer.train_sgd()

    # Decision Tree
    decision_tree = trainer.train_decision_tree()

    # Random Forest
    random_forest_pred = trainer.train_random_forest()

    # list all scores in dataframe
    scores_df = trainer.model_scores_to_dataframe()
    print(scores_df.sort_values(by="Score", ascending=False))
    # plot_model_scores(scores_df)

    # submit solution
    submit_results(test_df, best_model_predictions=random_forest_pred)


if __name__ == "__main__":
    main()
