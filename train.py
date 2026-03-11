"""
STEP 3: Training
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import optuna
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pre_processing import preprocess

RANDOM_STATE = 42

def train(train_scaled, val_scaled=None):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("SpaceshipTitanic")

    
    if train_scaled is None:
        train_df, val_df = preprocess()
    else:
        train_df, val_df = train_scaled, val_scaled

    X_train = train_df.drop("Transported", axis=1)
    y_train = train_df["Transported"]
    X_val = val_df.drop("Transported", axis=1)
    y_val = val_df["Transported"]
    
    X = pd.concat([X_train, X_val], axis=0)
    y = pd.concat([y_train, y_val], axis=0)

    # Fungsi objective Optuna
    def objective_lr(trial):
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 2000),
            'random_state': RANDOM_STATE
        }
        
        if params['solver'] == 'liblinear' and params['penalty'] not in ['l1', 'l2']:
            params['penalty'] = 'l2'
        
        model = LogisticRegression(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores.mean()

    print("Optimizing Logistic Regression...")
    from optuna.samplers import TPESampler
    sampler = TPESampler(seed=RANDOM_STATE)

    study_lr = optuna.create_study(direction='maximize', study_name='logistic_regression_optimization', sampler=sampler)
    study_lr.optimize(objective_lr, n_trials=30, show_progress_bar=True)

    # Training final & MLflow logging
    with mlflow.start_run() as run:
        print("Training Logistic Regression Baseline...")
        lr_baseline = LogisticRegression(random_state=RANDOM_STATE,max_iter=100)
        lr_baseline.fit(X_train, y_train)
        lr_pred = lr_baseline.predict(X_val)
        lr_acc = accuracy_score(y_val, lr_pred)
        mlflow.log_metric("baseline_accuracy", lr_acc)
        print(f"Logistic Regression Baseline Accuracy: {lr_acc:.4f}\n")


        print("Training Logistic Regression...")
        best_params = study_lr.best_params
        model = LogisticRegression(**best_params, random_state=RANDOM_STATE)
        model.fit(X, y)

        # Log parameter ke MLflow
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        mlflow.log_metric("cv_accuracy", study_lr.best_value)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name="spaceship_titanic_lr"
        )
        
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")
        print(f"Model trained. Run ID: {run.info.run_id}")
        return run.info.run_id

if __name__ == "__main__":
    train(None)