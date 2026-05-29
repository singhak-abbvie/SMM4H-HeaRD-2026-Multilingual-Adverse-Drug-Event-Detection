"""
Ensemble test set prediction — averages probabilities from multiple XLM-R-large models.

Usage:
    py predict_test_ensemble.py --test_file test_data.csv --output predictions.csv
"""

import argparse
import os
import re
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIRS = [
    "fine_tuned_models/xlmr_large_ade",       # XLM-R-large seed 42
    "fine_tuned_models/xlmr_large_ade_s123",  # XLM-R-large seed 123
    "fine_tuned_models/xlmr_large_ade_s456",  # XLM-R-large seed 456
]
MAX_LENGTH = 256
BATCH_SIZE = 32
THRESHOLDS_FILE = "ensemble_output/thresholds.json"


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'@USER[_]+\d*', '@USER', text)
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'<user>', '@USER', text)
    text = re.sub(r'<pi>', '[PLACE]', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def predict(model, texts, tokenizer, device):
    dataset = SimpleDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True, help="Path to test CSV (must have id, text, language)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--thresholds", default=THRESHOLDS_FILE, help="Path to ensemble thresholds JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load thresholds
    with open(args.thresholds) as f:
        thresholds = json.load(f)
    print(f"Thresholds: {thresholds}")
    default_threshold = np.mean(list(thresholds.values()))

    # Load test data
    test_df = pd.read_csv(args.test_file)
    test_df["text"] = test_df["text"].apply(normalize_text)
    print(f"Test data: {len(test_df)} samples")
    print(f"Languages: {test_df['language'].value_counts().to_dict()}")

    # Get probabilities from each model
    available_dirs = [d for d in MODEL_DIRS if os.path.isdir(d)]
    print(f"\nUsing {len(available_dirs)} models: {available_dirs}")

    all_probs = []
    for model_dir in available_dirs:
        print(f"\n  Loading {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()

        probs = predict(model, test_df["text"].tolist(), tokenizer, device)
        all_probs.append(probs)
        print(f"  {model_dir}: mean_prob={probs.mean():.4f}")

        del model
        torch.cuda.empty_cache()

    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    print(f"\nEnsemble avg prob: mean={avg_probs.mean():.4f}, std={avg_probs.std():.4f}")

    # Apply per-language thresholds
    predictions = np.zeros(len(avg_probs), dtype=int)
    for lang in test_df["language"].unique():
        mask = test_df["language"].values == lang
        thresh = thresholds.get(lang, default_threshold)
        predictions[mask] = (avg_probs[mask] >= thresh).astype(int)
        pred_pos = int(predictions[mask].sum())
        total = int(mask.sum())
        print(f"  {lang}: threshold={thresh}, predicted_pos={pred_pos}/{total} ({pred_pos/total:.1%})")

    # Save
    result = pd.DataFrame({"id": test_df["id"], "predicted_label": predictions})
    result.to_csv(args.output, index=False)
    print(f"\nSaved {len(result)} predictions to {args.output}")
    print(f"Total positive: {predictions.sum()} / {len(predictions)} ({predictions.mean():.1%})")


if __name__ == "__main__":
    main()
