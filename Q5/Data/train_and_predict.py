import os
import sys
import argparse
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score


warnings.filterwarnings("ignore", category=UserWarning)


def read_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Ensure the file exists or pass the correct path with --train"
        )
    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Test file not found: {test_path}\n"
            "Ensure the file exists or pass the correct path with --test"
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def resolve_data_paths(
    project_root: str,
    explicit_train: str | None,
    explicit_test: str | None,
    data_dir: str | None,
) -> tuple[str, str]:
    # If explicit files provided, prefer them
    if explicit_train and explicit_test:
        return explicit_train, explicit_test

    candidate_dirs: list[str] = []
    if data_dir:
        candidate_dirs.append(data_dir)
    candidate_dirs.extend([
        os.path.join(project_root, "Data"),
        os.path.join(project_root, "data"),
        project_root,
    ])

    # Check common candidate directories
    for candidate_dir in candidate_dirs:
        train_candidate = os.path.join(candidate_dir, "train.csv")
        test_candidate = os.path.join(candidate_dir, "test.csv")
        if os.path.isfile(train_candidate) and os.path.isfile(test_candidate):
            return train_candidate, test_candidate

    # Fallback: recursive search across project_root
    train_found: list[str] = []
    test_found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(project_root):
        if "train.csv" in filenames:
            train_found.append(os.path.join(dirpath, "train.csv"))
        if "test.csv" in filenames:
            test_found.append(os.path.join(dirpath, "test.csv"))

    def pick_best(paths: list[str]) -> str | None:
        if not paths:
            return None
        # Prefer paths under a directory named like 'data'
        preferred = [p for p in paths if os.path.basename(os.path.dirname(p)).lower() in {"data", "dataset", "datasets"}]
        selected = preferred[0] if preferred else min(paths, key=len)
        return selected

    train_path = pick_best(train_found)
    test_path = pick_best(test_found)
    if train_path and test_path:
        return train_path, test_path

    missing_hints = [
        f"Checked: {os.path.join(project_root, 'Data')} and {os.path.join(project_root, 'data')} and {project_root}",
        "You can pass --data-dir /absolute/path/to/dir or --train/--test with absolute paths.",
    ]
    raise FileNotFoundError(
        "Could not locate train.csv and/or test.csv under the project.\n" + "\n".join(missing_hints)
    )


def combine_headlines(df: pd.DataFrame, headline_cols: List[str]) -> pd.Series:
    # Ensure text type and safe join
    texts = (
        df[headline_cols]
        .fillna("")
        .astype(str)
        .agg(". ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return texts


def prepare_texts(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    headline_cols = [f"Top{i}" for i in range(1, 26)]
    # Some datasets may use spaces like 'Top1' exactly; already confirmed header names
    X_train_text = combine_headlines(train_df, headline_cols)
    X_test_text = combine_headlines(test_df, headline_cols)
    return X_train_text, X_test_text


def vectorize_texts(X_train_text: pd.Series, X_test_text: pd.Series):
    # Word-level TF-IDF
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=150000,
        min_df=2,
    )
    # Char-level TF-IDF (captures misspellings, byte-like prefixes, etc.)
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200000,
    )

    Xw = word_tfidf.fit_transform(X_train_text)
    Xt_w = word_tfidf.transform(X_test_text)

    Xc = char_tfidf.fit_transform(X_train_text)
    Xt_c = char_tfidf.transform(X_test_text)

    from scipy.sparse import hstack

    X_train = hstack([Xw, Xc]).tocsr()
    X_test = hstack([Xt_w, Xt_c]).tocsr()
    return X_train, X_test


def time_series_cv_train(X, y, n_splits: int = 5, random_state: int = 42) -> SGDClassifier:
    # Linear SVM via SGD with hinge loss; class_weight to handle imbalance
    best_model: SGDClassifier | None = None
    best_score = -1.0

    splitter = TimeSeriesSplit(n_splits=n_splits)

    # Try a few alphas
    alphas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

    for alpha in alphas:
        fold_scores: List[float] = []
        for train_idx, val_idx in splitter.split(X):
            clf = SGDClassifier(
                loss="hinge",
                alpha=alpha,
                penalty="l2",
                max_iter=2000,
                tol=1e-3,
                random_state=random_state,
                class_weight="balanced",
            )
            clf.fit(X[train_idx], y[train_idx])
            val_pred = clf.predict(X[val_idx])
            score = f1_score(y[val_idx], val_pred, average="weighted")
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores)) if fold_scores else -1.0
        if mean_score > best_score:
            best_score = mean_score
            best_model = SGDClassifier(
                loss="hinge",
                alpha=alpha,
                penalty="l2",
                max_iter=4000,
                tol=1e-3,
                random_state=random_state,
                class_weight="balanced",
            )

    assert best_model is not None
    best_model.fit(X, y)
    return best_model


def main():
    parser = argparse.ArgumentParser(description="Train model and predict labels for news data")
    parser.add_argument("--train", dest="train_csv", type=str, default=None, help="Path to train.csv")
    parser.add_argument("--test", dest="test_csv", type=str, default=None, help="Path to test.csv")
    parser.add_argument("--out", dest="out_csv", type=str, default=None, help="Path to output submission.csv")
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=None, help="Directory containing train.csv and test.csv")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    # Defaults assume Q5/Data structure
    default_train_csv = os.path.join(project_root, "Data", "train.csv")
    default_test_csv = os.path.join(project_root, "Data", "test.csv")
    out_dir = os.path.join(project_root, "notebook")
    os.makedirs(out_dir, exist_ok=True)
    default_out_csv = os.path.join(out_dir, "submission.csv")

    # Resolve data paths robustly
    train_csv, test_csv = resolve_data_paths(
        project_root=project_root,
        explicit_train=args.train_csv,
        explicit_test=args.test_csv,
        data_dir=args.data_dir,
    )
    out_csv = args.out_csv or default_out_csv

    train_df, test_df = read_data(train_csv, test_csv)

    # Sort by Date to respect temporal order
    train_df = train_df.sort_values("Date").reset_index(drop=True)
    test_df = test_df.sort_values("Date").reset_index(drop=True)

    # Extract target
    if "Label" not in train_df.columns:
        raise ValueError("Label column not found in training data")
    y = train_df["Label"].astype(int).to_numpy()

    X_train_text, X_test_text = prepare_texts(train_df, test_df)
    X_train, X_test = vectorize_texts(X_train_text, X_test_text)

    model = time_series_cv_train(X_train, y)

    test_pred = model.predict(X_test)
    # Ensure 0/1 ints
    test_pred = np.asarray(test_pred, dtype=int)

    submission = pd.DataFrame({"prediction": test_pred})
    submission.to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)


