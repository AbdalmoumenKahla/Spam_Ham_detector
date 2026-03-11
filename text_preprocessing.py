import re
from pathlib import Path

import pandas as pd

VALID_LABELS = {"ham", "spam"}
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
	text = str(text).lower()
	text = URL_RE.sub(" url ", text)
	text = EMAIL_RE.sub(" email ", text)
	text = NON_WORD_RE.sub(" ", text)
	return WHITESPACE_RE.sub(" ", text).strip()


def load_dataset(csv_path: str = "spam_clean.csv") -> pd.DataFrame:
	dataset_path = Path(csv_path)
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset not found: {dataset_path}")

	dataset = pd.read_csv(dataset_path)
	required_columns = {"label", "text"}
	missing_columns = required_columns.difference(dataset.columns)
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Dataset is missing required columns: {missing}")

	dataset = dataset.loc[:, ["label", "text"]].copy()
	dataset["label"] = dataset["label"].astype(str).str.strip().str.lower()
	dataset["text"] = dataset["text"].astype(str).map(clean_text)

	dataset = dataset[dataset["label"].isin(VALID_LABELS)]
	dataset = dataset[dataset["text"].str.len() > 0]
	dataset = dataset.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
	return dataset