"""
Rank features by importance to guide feature selection.
Combines three signals:
  1. Random Forest impurity-based importance
  2. Permutation importance (model-agnostic, more reliable)
  3. Mutual information (non-linear univariate score)
"""
import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

FEAST_REPO_PATH = "feature_repo/feature_repo"
SAVED_DATASET_NAME = "training_dataset"
TARGET = "satisfaction"
DROP_COLS = ["satisfaction", "passenger_id", "event_timestamp"]


def load_data():
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    df = store.get_saved_dataset(SAVED_DATASET_NAME).to_df()
    y = df[TARGET]
    X = df.drop(columns=DROP_COLS, axis=1)
    return train_test_split(X, y, stratify=y, random_state=42)


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Features: {X_train.shape[1]}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 1. Random Forest impurity importance
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"\nReference RF accuracy (all features): {rf_acc:.4f}")

    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns)

    # 2. Permutation importance on the test set
    print("\nComputing permutation importance...")
    perm = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=X_train.columns)

    # 3. Mutual information
    print("Computing mutual information...")
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    mi_imp = pd.Series(mi, index=X_train.columns)

    # Combine into a ranking table
    ranks = pd.DataFrame({
        "rf_importance": rf_imp,
        "permutation_importance": perm_imp,
        "mutual_info": mi_imp,
    })
    ranks["rf_rank"] = ranks["rf_importance"].rank(ascending=False).astype(int)
    ranks["perm_rank"] = ranks["permutation_importance"].rank(ascending=False).astype(int)
    ranks["mi_rank"] = ranks["mutual_info"].rank(ascending=False).astype(int)
    ranks["avg_rank"] = ranks[["rf_rank", "perm_rank", "mi_rank"]].mean(axis=1)
    ranks = ranks.sort_values("avg_rank")

    print("\n=== Feature Ranking (lower avg_rank = more important) ===")
    print(ranks.to_string(float_format=lambda x: f"{x:.4f}"))

    ranks.to_csv("feature_ranking.csv")
    print("\nSaved: feature_ranking.csv")

    # Suggest top-K subsets and evaluate
    print("\n=== Accuracy vs. top-K features (RF) ===")
    ordered = ranks.index.tolist()
    for k in [5, 10, 15, 20, len(ordered)]:
        cols = ordered[:k]
        m = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        m.fit(X_train[cols], y_train)
        acc = m.score(X_test[cols], y_test)
        print(f"  top-{k:>2}: accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
