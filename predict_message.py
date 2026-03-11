import argparse
from pathlib import Path

from typing import Any

import joblib

DEFAULT_MODEL_PATH = Path("artifacts/spam_classifier.joblib")


def predict_text(message: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
	model = joblib.load(model_path)
	prediction = model.predict([message])[0]
	result: dict[str, Any] = {"prediction": prediction}

	if hasattr(model, "predict_proba"):
		probabilities = model.predict_proba([message])[0]
		result["probabilities"] = {
			label: float(probability)
			for label, probability in zip(model.classes_, probabilities)
		}

	return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether a message is spam or ham.")
    parser.add_argument("message", help="The SMS message to classify.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model file.",
    )
    args = parser.parse_args()

    result = predict_text(args.message, args.model)
    print(f"Prediction: {result['prediction']}")

    for label, probability in sorted(result.get("probabilities", {}).items()):
        print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()