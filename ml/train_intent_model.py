import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Example:
    text: str
    intent: str


def _read_jsonl(path: str) -> List[Example]:
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            text = str((obj or {}).get("text") or "").strip()
            intent = str((obj or {}).get("intent") or "").strip()
            if not text or not intent:
                raise SystemExit(f"{path}:{line_no}: each row needs 'text' and 'intent'")
            examples.append(Example(text=text, intent=intent))
    if not examples:
        raise SystemExit(f"{path}: no training examples found")
    return examples


def _build_pipeline() -> Pipeline:
    # Char n-grams make this work reasonably well across Indian languages without tokenizers.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=1,
    )
    clf = LogisticRegression(
        max_iter=4000,
        n_jobs=None,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train(examples: Iterable[Example], test_size: float, seed: int) -> Tuple[Pipeline, str]:
    texts = [ex.text for ex in examples]
    labels = [ex.intent for ex in examples]

    if len(set(labels)) < 2:
        raise SystemExit("Need at least 2 distinct intents to train a classifier")

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    pipe = _build_pipeline()
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    report = classification_report(y_test, y_pred, digits=3)
    return pipe, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an intent classifier for Agri Mitra voice assistant")
    parser.add_argument("--data", default=os.path.join("ml", "intent_data.jsonl"))
    parser.add_argument("--out", default=os.path.join("api", "models", "intent.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = _read_jsonl(args.data)
    pipe, report = train(examples, test_size=args.test_size, seed=args.seed)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipe,
            "trained_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "data_path": args.data,
        },
        args.out,
    )

    print("Saved:", args.out)
    print(report)


if __name__ == "__main__":
    main()

