import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    X_train = pd.read_csv("titanic_preprocessing/X_train_processed.csv")
    X_test  = pd.read_csv("titanic_preprocessing/X_test_processed.csv")
    y_train = pd.read_csv("titanic_preprocessing/y_train.csv").squeeze()
    y_test  = pd.read_csv("titanic_preprocessing/y_test.csv").squeeze()

    # AUTLOG saja, jangan start_run manual
    mlflow.sklearn.autolog()

    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()
