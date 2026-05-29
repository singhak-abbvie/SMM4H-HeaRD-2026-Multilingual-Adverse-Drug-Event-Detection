"""
Ensemble: Average probabilities from multiple XLM-R-large models (different seeds).
Also supports re-extracting probabilities from saved model checkpoints.

Usage:
    py ensemble_seeds.py
    py ensemble_seeds.py --reextract   # re-run inference from model checkpoints
"""

import os
import json
import argparse
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIRS = [
    "fine_tuned_models/xlmr_large_ade",       # XLM-R-large seed 42
    "fine_tuned_models/xlmr_large_ade_s123",  # XLM-R-large seed 123
    "fine_tuned_models/xlmr_large_ade_s456",  # XLM-R-large seed 456
]

DEV_MAIN = "training_datasets/dev_data_SMM4H_2026_Task_1.csv"
DEV_CADEC = "training_datasets/dev_data_cadec_translated.csv"
MAX_LENGTH = 256
BATCH_SIZE = 32


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


class ADEDataset(Dataset):
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
def extract_probs(model, dataloader, device):
    model.eval()
    all_probs = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def find_best_threshold(y_true, probs, low=0.2, high=0.8, step=0.01):
    best_f1, best_t = 0, 0.5
    for t in np.arange(low, high, step):
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate_ensemble(avg_probs, dev_df, label=""):
    labels_true = dev_df["label"].astype(int).values
    languages = dev_df["language"].values

    lang_thresholds = {}
    final_preds = np.zeros(len(avg_probs), dtype=int)

    print(f"\n  {label} Per-language results:")
    for lang in sorted(set(languages)):
        mask = languages == lang
        lang_true = labels_true[mask]
        lang_probs = avg_probs[mask]
        lang_indices = np.where(mask)[0]

        t, f1 = find_best_threshold(lang_true, lang_probs)
        lang_thresholds[lang] = t
        lang_preds = (lang_probs >= t).astype(int)
        final_preds[lang_indices] = lang_preds

        prec, rec, _, _ = precision_recall_fscore_support(
            lang_true, lang_preds, average="binary", zero_division=0
        )
        print(f"    {lang}: thresh={t:.2f} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    _, _, overall_f1, _ = precision_recall_fscore_support(
        labels_true, final_preds, average="binary", zero_division=0
    )
    print(f"  {label} Overall F1 = {overall_f1:.4f}")

    return final_preds, lang_thresholds, overall_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reextract", action="store_true",
                        help="Re-run inference from model checkpoints instead of using saved probs")
    args = parser.parse_args()

    dev_main = pd.read_csv(DEV_MAIN)
    dev_cadec = pd.read_csv(DEV_CADEC)
    dev_main["text"] = dev_main["text"].apply(normalize_text)
    dev_cadec["text"] = dev_cadec["text"].apply(normalize_text)

    available_dirs = [d for d in MODEL_DIRS if os.path.isdir(d)]
    print(f"Found {len(available_dirs)} model directories: {available_dirs}")

    if len(available_dirs) < 2:
        print("ERROR: Need at least 2 models for ensemble. Train more seeds first.")
        return

    # ── Collect probabilities ───────────────────────────────────────────────
    all_main_probs = []
    all_cadec_probs = []

    if args.reextract:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Re-extracting probabilities from checkpoints (device: {device})")

        for model_dir in available_dirs:
            print(f"\n  Loading {model_dir}...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

            # Main dev
            ds = ADEDataset(dev_main["text"].tolist(), tokenizer, MAX_LENGTH)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            probs = extract_probs(model, loader, device)
            all_main_probs.append(probs)

            # CADEC dev
            ds = ADEDataset(dev_cadec["text"].tolist(), tokenizer, MAX_LENGTH)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            probs = extract_probs(model, loader, device)
            all_cadec_probs.append(probs)

            # Free memory
            del model
            torch.cuda.empty_cache()
    else:
        for model_dir in available_dirs:
            main_probs_file = os.path.join(model_dir, "dev_probs_best.csv")
            cadec_probs_file = os.path.join(model_dir, "dev_cadec_probs_best.csv")

            if not os.path.exists(main_probs_file):
                print(f"WARNING: {main_probs_file} not found — skipping {model_dir}")
                continue

            main_probs_df = pd.read_csv(main_probs_file)
            all_main_probs.append(main_probs_df["prob"].values)
            print(f"  {model_dir}: loaded main probs ({len(main_probs_df)} samples)")

            if os.path.exists(cadec_probs_file):
                cadec_probs_df = pd.read_csv(cadec_probs_file)
                all_cadec_probs.append(cadec_probs_df["prob"].values)
                print(f"  {model_dir}: loaded CADEC probs ({len(cadec_probs_df)} samples)")

    n_models = len(all_main_probs)
    print(f"\nEnsembling {n_models} models...")

    # ── Average probabilities ───────────────────────────────────────────────
    avg_main_probs = np.mean(all_main_probs, axis=0)
    print(f"\n{'='*60}")
    print(f"MAIN DEV ({n_models}-model ensemble)")
    print(f"{'='*60}")
    preds_main, thresholds_main, f1_main = evaluate_ensemble(avg_main_probs, dev_main, "Ensemble")

    # Save ensemble predictions
    pd.DataFrame({"id": dev_main["id"], "predicted_label": preds_main}).to_csv(
        "predicted_datasets/predictions_ensemble_seed_dev.csv", index=False
    )
    # Save ensemble probs
    pd.DataFrame({"id": dev_main["id"], "prob": avg_main_probs}).to_csv(
        "predicted_datasets/ensemble_avg_probs_main.csv", index=False
    )

    # Official scoring
    print(f"\nOfficial scoring (main):")
    os.system('py scoring_task1.py --goldstandard_file "training_datasets/dev_data_SMM4H_2026_Task_1.csv" '
              '--predictions_file "predicted_datasets/predictions_ensemble_seed_dev.csv"')

    # CADEC
    if all_cadec_probs:
        avg_cadec_probs = np.mean(all_cadec_probs, axis=0)
        print(f"\n{'='*60}")
        print(f"CADEC DEV ({n_models}-model ensemble)")
        print(f"{'='*60}")
        preds_cadec, thresholds_cadec, f1_cadec = evaluate_ensemble(avg_cadec_probs, dev_cadec, "Ensemble")

        pd.DataFrame({"id": dev_cadec["id"], "predicted_label": preds_cadec}).to_csv(
            "predicted_datasets/predictions_ensemble_seed_cadec_dev.csv", index=False
        )

        print(f"\nOfficial scoring (CADEC):")
        os.system('py scoring_task1.py --goldstandard_file "training_datasets/dev_data_cadec_translated.csv" '
                  '--predictions_file "predictions_ensemble_seed_cadec_dev.csv"')

    # ── Save thresholds ─────────────────────────────────────────────────────
    os.makedirs("ensemble_output", exist_ok=True)
    with open("ensemble_output/thresholds.json", "w") as f:
        json.dump({k: round(float(v), 2) for k, v in thresholds_main.items()}, f, indent=2)

    # ── Individual model comparison ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Individual model F1s vs Ensemble:")
    print(f"{'='*60}")
    for i, model_dir in enumerate(available_dirs[:n_models]):
        probs_i = all_main_probs[i]
        labels_true = dev_main["label"].astype(int).values
        languages = dev_main["language"].values
        final_preds = np.zeros(len(probs_i), dtype=int)
        lang_f1s = {}
        for lang in sorted(set(languages)):
            mask = languages == lang
            t, f1 = find_best_threshold(labels_true[mask], probs_i[mask])
            lang_f1s[lang] = f1
            final_preds[np.where(mask)[0]] = (probs_i[mask] >= t).astype(int)
        _, _, ov_f1, _ = precision_recall_fscore_support(labels_true, final_preds, average="binary", zero_division=0)
        lang_str = " ".join(f"{l}={f:.3f}" for l, f in sorted(lang_f1s.items()))
        print(f"  {model_dir}: F1={ov_f1:.4f}  ({lang_str})")

    print(f"\n  ENSEMBLE:  F1={f1_main:.4f}")
    print(f"\nDone! Predictions saved to predictions_ensemble_seed_dev.csv")


if __name__ == "__main__":
    main()
