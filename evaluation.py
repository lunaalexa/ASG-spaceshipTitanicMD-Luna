"""
STEP 4: Evaluation
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pre_processing import preprocess

def evaluate(val_scaled, run_id):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    if val_scaled is None:
        print("Loading data from preprocess...")
        _, val_df = preprocess()
    else:
        val_df = val_scaled

    X_test = val_df.drop("Transported", axis=1)
    y_test = val_df["Transported"]

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")


    preds = model.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="binary")
    rec  = recall_score(y_test, preds, average="binary")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)

    print(f"Evaluation | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")
    return acc, prec, rec


if __name__ == "__main__":
    evaluate(None, None)