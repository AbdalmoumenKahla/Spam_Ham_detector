# Spam / Ham Detector

A machine-learning pipeline that classifies SMS messages as **spam** or **ham** (legitimate), served via a browser-based test interface.

---

## How it works

1. Raw SMS data (`spam.csv`) is cleaned by `text_preprocessing.py`.
2. `train_model.py` builds a **TF-IDF + Logistic Regression** pipeline, evaluates it, and saves the trained model to `artifacts/spam_classifier.joblib`.
3. `app.py` spins up a local HTTP server that exposes a REST prediction endpoint and serves the browser UI.
4. The browser UI (`ui/`) lets you type or paste any message and see the prediction with confidence scores in real time.

---

## Project structure

```
├── text_preprocessing.py        # Text cleaning helpers and dataset loader
├── train_model.py               # Model training, evaluation, and saving
├── predict_message.py           # CLI tool + reusable predict_text() function
├── app.py                       # Local HTTP server (serves UI + /api/predict)
├── ui/
│   ├── index.html               # Browser test interface
│   ├── styles.css               # Styling
│   └── app.js                   # Fetch logic and result rendering
├── artifacts/
│   └── spam_classifier.joblib   # Saved trained model
├── spam_clean.csv               # Preprocessed dataset (label, text)
└── .gitignore
```

---

## Requirements

- Python 3.10+
- `pandas`
- `scikit-learn`
- `joblib`

Install dependencies into a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install pandas scikit-learn joblib
```

---

## Usage

### 1 — Train the model

```bash
python train_model.py
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--data` | `spam_clean.csv` | Path to the cleaned CSV |
| `--model-out` | `artifacts/spam_classifier.joblib` | Where to save the model |
| `--test-size` | `0.2` | Fraction of data used for testing |
| `--random-state` | `42` | Random seed for reproducibility |

Example output:

```
Rows used: 5169
Train rows: 4135
Test rows: 1034
Accuracy: 0.9835
Classification report:
              precision    recall  f1-score   support
         ham     0.9862    0.9938    0.9900       966
        spam     0.9590    0.9143    0.9361        70
...
Saved model to: artifacts/spam_classifier.joblib
```

---

### 2 — Run the web interface

```bash
python app.py
```

Then open **http://127.0.0.1:8000** in your browser.

- Paste any SMS message and click **Analyze message** (or press `Ctrl+Enter`).
- Use the **Spam sample** / **Ham sample** buttons to try built-in examples.
- Confidence scores for both classes are displayed as percentage bars.

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Host to bind |
| `--port` | `8000` | Port to bind |

---

### 3 — Command-line prediction

```bash
python predict_message.py "Free entry! Win a prize. Text CLAIM to 80888."
```

Output:

```
Prediction: spam
ham: 0.0351
spam: 0.9649
```

---

## Dataset

The [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from UCI via Kaggle.

- **5,572** raw messages → **5,169** after deduplication and cleaning
- **ham**: 4,825 messages (~86.6%)
- **spam**: 747 messages (~13.4%)

The `spam_clean.csv` file contains two columns: `label` (`ham` or `spam`) and `text`.

---

## Model details

| Component | Choice |
|---|---|
| Vectorizer | `TfidfVectorizer` (unigrams + bigrams, `sublinear_tf=True`) |
| Classifier | `LogisticRegression` (`class_weight="balanced"`, `max_iter=1000`) |
| Train/test split | 80 / 20 stratified |

Class weighting is applied to compensate for the imbalanced dataset (~87% ham vs ~13% spam).

---

## API reference

### `POST /api/predict`

**Request body (JSON):**

```json
{ "message": "Your text here" }
```

**Response (JSON):**

```json
{
  "prediction": "spam",
  "probabilities": {
    "ham": 0.0351,
    "spam": 0.9649
  }
}
```
