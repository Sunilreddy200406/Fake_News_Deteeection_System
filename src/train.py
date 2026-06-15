from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.preprocess import ensure_nltk_resources, preprocess_text


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "docs" / "figures"


def load_dataset() -> pd.DataFrame:
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"
    sample_path = DATA_DIR / "sample_fake_news.csv"

    if fake_path.exists() and true_path.exists():
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        fake_df["label"] = 0
        true_df["label"] = 1
        df = pd.concat([fake_df, true_df], ignore_index=True)
        text_col = "text" if "text" in df.columns else df.columns[0]
        title_col = "title" if "title" in df.columns else None
        if title_col:
            df["combined_text"] = (df[title_col].fillna("") + " " + df[text_col].fillna("")).str.strip()
        else:
            df["combined_text"] = df[text_col].fillna("")
        df = df[["combined_text", "label"]].rename(columns={"combined_text": "text"})
        return df.dropna()

    if sample_path.exists():
        df = pd.read_csv(sample_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("sample_fake_news.csv must contain 'text' and 'label' columns.")
        df["label"] = df["label"].map(
            {
                "FAKE": 0,
                "REAL": 1,
                "fake": 0,
                "real": 1,
                0: 0,
                1: 1,
            }
        )
        return df[["text", "label"]].dropna()

    raise FileNotFoundError(
        "Dataset not found. Add Kaggle files data/Fake.csv and data/True.csv "
        "or use data/sample_fake_news.csv."
    )


def run_eda(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    plt.figure(figsize=(6, 4))
    counts = df["label"].map({0: "Fake", 1: "Real"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="Set2", legend=False)
    plt.title("Class Distribution")
    plt.xlabel("News Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png")
    plt.close()

    eda_df = df.copy()
    eda_df["text_len"] = eda_df["text"].str.len()
    plt.figure(figsize=(8, 4))
    sns.histplot(data=eda_df, x="text_len", hue="label", bins=40, kde=True, element="step")
    plt.title("Text Length Distribution")
    plt.xlabel("Characters")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "text_length_distribution.png")
    plt.close()


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, model_key: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {model_key}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"cm_{model_key}.png")
    plt.close()


def train_and_compare(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = df.copy()
    print("Preprocessing text. This can take a few minutes on full Kaggle data...")
    df["processed_text"] = df["text"].astype(str).apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    models = {
        "logistic_regression": LogisticRegression(max_iter=300, random_state=42),
        "naive_bayes": MultinomialNB(),
    }

    results = []
    best_bundle: Dict[str, object] = {}
    best_f1 = -1.0

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        model_key = f"tfidf_{model_name}"
        metrics = evaluate_model(y_test, preds)
        metrics.update({"vectorizer": "tfidf", "model": model_name})
        results.append(metrics)
        save_confusion_matrix(y_test, preds, model_key)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_bundle = {
                "vectorizer_name": "tfidf",
                "model_name": model_name,
                "vectorizer": vectorizer,
                "model": model,
                "label_map": {0: "Fake", 1: "Real"},
            }

        print(
            f"{model_key}: "
            f"acc={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1_score']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    return results_df, best_bundle


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ensure_nltk_resources()

    df = load_dataset()
    print(f"Loaded dataset shape: {df.shape}")
    run_eda(df)
    print(f"EDA plots saved to: {FIGURES_DIR}")

    results_df, best_bundle = train_and_compare(df)
    results_path = MODELS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    best_path = MODELS_DIR / "best_model.pkl"
    with open(best_path, "wb") as f:
        pickle.dump(best_bundle, f)

    print("\nModel comparison (top 6):")
    print(results_df.head(6).to_string(index=False))
    print(f"\nSaved comparison file: {results_path}")
    print(f"Saved best model bundle: {best_path}")
    print(
        f"Best model: {best_bundle['vectorizer_name']} + {best_bundle['model_name']} "
        f"(F1={results_df.iloc[0]['f1_score']:.4f})"
    )


if __name__ == "__main__":
    main()
