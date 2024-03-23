import numpy as np
import pandas as pd
from sklearn.calibration import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Trainer:

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.X_train = train_df.drop("Survived", axis=1)
        self.Y_train = train_df["Survived"]
        self.X_test = test_df.drop("PassengerId", axis=1).copy()
        self.scores = dict()

    def train_lr(self):
        # Logistic Regression
        logreg = LogisticRegression()
        logreg.fit(self.X_train, self.Y_train)
        Y_pred = logreg.predict(self.X_test)
        acc_log = round(logreg.score(self.X_train, self.Y_train) * 100, 2)
        print(acc_log)
        self.scores["Logistic Regression"] = acc_log
        return logreg

    def correlation_to_lr(
        self, df: pd.DataFrame, logreg: LogisticRegression
    ) -> pd.DataFrame:
        coeff_df = pd.DataFrame(df.columns.delete(0))
        coeff_df.columns = ["Feature"]
        coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
        return coeff_df

    def train_svm(self):
        # Support Vector Machines
        svc = SVC()
        svc.fit(self.X_train, self.Y_train)
        Y_pred = svc.predict(self.X_test)
        acc_svc = round(svc.score(self.X_train, self.Y_train) * 100, 2)
        self.scores["Support Vector Machines"] = acc_svc
        print(f"svm accuracy: {acc_svc}")

    def train_knn(self, neighbors: int):
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(self.X_train, self.Y_train)
        Y_pred = knn.predict(self.X_test)
        acc_knn = round(knn.score(self.X_train, self.Y_train) * 100, 2)
        self.scores["KNN"] = acc_knn
        print(f"knn classifier accuracy: {acc_knn}")

    def train_naive_bayes(self):
        gaussian = GaussianNB()
        gaussian.fit(self.X_train, self.Y_train)
        Y_pred = gaussian.predict(self.X_test)
        acc_gaussian = round(gaussian.score(self.X_train, self.Y_train) * 100, 2)
        self.scores["Naive Bayes"] = acc_gaussian
        print(f"gaussian accuracy: {acc_gaussian}")

    def train_perceptron(self):
        perceptron = Perceptron()
        perceptron.fit(self.X_train, self.Y_train)
        Y_pred = perceptron.predict(self.X_test)
        acc_perceptron = round(perceptron.score(self.X_train, self.Y_train) * 100, 2)
        self.scores["Perceptron"] = acc_perceptron
        print(f"perceptron accuracy: {acc_perceptron}")

    def train_linear_svc(self):
        linear_svc = LinearSVC(dual="auto")
        linear_svc.fit(self.X_train, self.Y_train)
        Y_pred = linear_svc.predict(self.X_test)
        acc_linear_svc = round(linear_svc.score(self.X_train, self.Y_train) * 100, 2)
        self.scores["Linear SVC"] = acc_linear_svc
        print(f"linear svc accuracy: {acc_linear_svc}")

    def train_sgd(self):
        # Stochastic Gradient Descent
        sgd = SGDClassifier()
        sgd.fit(self.X_train, self.Y_train)
        Y_pred = sgd.predict(self.X_test)
        acc_sgd = round(sgd.score(self.X_train, self.Y_train) * 100, 2)
        self.scores["Stochastic Gradient Decent"] = acc_sgd
        print(f"sgd accuracy: {acc_sgd}")

    def train_decision_tree(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.X_train, self.Y_train)
        Y_pred = decision_tree.predict(self.X_test)
        acc_decision_tree = round(
            decision_tree.score(self.X_train, self.Y_train) * 100, 2
        )
        self.scores["Decision Tree"] = acc_decision_tree
        print(f"decision tree accuracy: {acc_decision_tree}")

    def train_random_forest(self) -> np.ndarray:
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(self.X_train, self.Y_train)
        Y_pred = random_forest.predict(self.X_test)
        random_forest.score(self.X_train, self.Y_train)
        acc_random_forest = round(
            random_forest.score(self.X_train, self.Y_train) * 100, 2
        )
        self.scores["Random Forest"] = acc_random_forest
        print(f"random forest accuracy: {acc_random_forest}")
        return Y_pred

    def model_scores_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.scores, orient="index", columns=["Score"])
