"""
Model training with MLflow tracking.
Uses the Feast training dataset prepared in feast.ipynb.
"""
import os
import joblib
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

FEAST_REPO_PATH = "feature_repo/feature_repo"
SAVED_DATASET_NAME = "training_dataset"
TARGET = "satisfaction"
DROP_COLS = ["satisfaction", "passenger_id", "event_timestamp"]
MODEL_DIR = "models"
EXPERIMENT_NAME = "airline_satisfaction"

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_training_data():
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    training_df = store.get_saved_dataset(SAVED_DATASET_NAME).to_df()
    y = training_df[TARGET]
    X = training_df.drop(columns=DROP_COLS, axis=1)
    return train_test_split(X, y, stratify=y, random_state=42)


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def train_and_log(name, model, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("model_type", name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        mlflow.log_param("feast_dataset", SAVED_DATASET_NAME)

        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, name="model")

        local_path = os.path.join(MODEL_DIR, f"model_{name}.joblib")
        joblib.dump(model, local_path)
        mlflow.log_artifact(local_path)

        print(f"[{name}] {metrics}")
        return metrics


def main():
    X_train, X_test, y_train, y_test = load_training_data()
    print(f"Loaded Feast training dataset: train={X_train.shape}, test={X_test.shape}")

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"max_iter": 1000, "random_state": 42},
        ),
        "decision_tree": (
            DecisionTreeClassifier(max_depth=10, random_state=42),
            {"max_depth": 10, "random_state": 42},
        ),
        "naive_bayes": (
            GaussianNB(),
            {"model": "GaussianNB"},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            {"n_estimators": 100, "max_depth": 15, "random_state": 42},
        ),
    }

    results = {}
    for name, (model, params) in models.items():
        results[name] = train_and_log(name, model, params, X_train, X_test, y_train, y_test)

    print("\n=== Summary ===")
    for name, m in results.items():
        print(f"{name}: acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
