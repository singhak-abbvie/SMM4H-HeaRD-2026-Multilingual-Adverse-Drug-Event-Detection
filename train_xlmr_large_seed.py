"""
XLM-RoBERTa-large Fine-tuning for ADE Detection — Multi-seed variant.
Usage:
    py train_xlmr_large_seed.py --seed 123 --output_dir fine_tuned_models/xlmr_large_ade_s123
    py train_xlmr_large_seed.py --seed 456 --output_dir fine_tuned_models/xlmr_large_ade_s456
"""

import os
import re
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import precision_recall_fscore_support

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "xlm-roberta-large"
MAX_LENGTH = 256
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
EPOCHS = 5
LR = 1e-5
WARMUP_RATIO = 0.1
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75
WEIGHT_DECAY = 0.01

TRAIN_MAIN = "training_datasets/train_data_SMM4H_2026_Task_1.csv"
TRAIN_CADEC = "training_datasets/train_data_cadec_translated.csv"
DEV_MAIN = "training_datasets/dev_data_SMM4H_2026_Task_1.csv"
DEV_CADEC = "training_datasets/dev_data_cadec_translated.csv"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
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
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, optimizer, scheduler, scaler, focal_loss, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = focal_loss(outputs.logits, labels)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        total_loss += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(dataloader)


@torch.no_grad()
def get_predictions(model, dataloader, device):
    model.eval()
    all_probs = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
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


def evaluate_dev(model, dev_df, tokenizer, device, epoch_label=""):
    dataset = ADEDataset(
        dev_df["text"].tolist(), dev_df["label"].astype(int).tolist(),
        tokenizer, MAX_LENGTH,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0)
    probs = get_predictions(model, loader, device)

    labels_true = dev_df["label"].astype(int).values
    languages = dev_df["language"].values

    lang_thresholds = {}
    final_preds = np.zeros(len(probs), dtype=int)

    for lang in sorted(set(languages)):
        mask = languages == lang
        lang_true = labels_true[mask]
        lang_probs = probs[mask]
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
    print(f"  {epoch_label} Overall F1: {overall_f1:.4f}")

    return probs, final_preds, lang_thresholds, overall_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    SEED = args.seed
    OUTPUT_DIR = args.output_dir

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {SEED}, Output: {OUTPUT_DIR}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load & Normalize Data ───────────────────────────────────────────────
    print("\nLoading data...")
    train_main = pd.read_csv(TRAIN_MAIN)
    train_cadec = pd.read_csv(TRAIN_CADEC)
    dev_main = pd.read_csv(DEV_MAIN)
    dev_cadec = pd.read_csv(DEV_CADEC)

    for df in [train_main, train_cadec, dev_main, dev_cadec]:
        df["text"] = df["text"].apply(normalize_text)

    train_df = pd.concat([train_main, train_cadec], ignore_index=True)
    train_df = train_df.dropna(subset=["label", "text"])
    train_df["label"] = train_df["label"].astype(int)

    pos_df = train_df[train_df["label"] == 1]
    train_df = pd.concat([train_df] + [pos_df] * 2, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Training: {len(train_df)} samples (after 3x oversampling)")

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    train_dataset = ADEDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LENGTH,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
    )

    num_training_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    scaler = torch.amp.GradScaler("cuda")
    focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    print(f"Training: {EPOCHS} epochs, effective batch={BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # ── Training ────────────────────────────────────────────────────────────
    best_f1 = 0
    best_thresholds = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, focal_loss, device)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{EPOCHS} — loss={avg_loss:.4f}, time={elapsed:.0f}s")

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak VRAM: {peak:.1f} GB")

        print("  Main dev:")
        probs_main, preds, thresholds, f1 = evaluate_dev(model, dev_main, tokenizer, device, f"[E{epoch}]")

        print("  CADEC dev:")
        probs_cadec, _, _, cadec_f1 = evaluate_dev(model, dev_cadec, tokenizer, device, f"[E{epoch}]")

        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = thresholds
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w") as fp:
                json.dump({k: float(v) for k, v in thresholds.items()}, fp, indent=2)
            # Save probabilities for ensemble
            pd.DataFrame({"id": dev_main["id"].values, "prob": probs_main}).to_csv(
                os.path.join(OUTPUT_DIR, "dev_probs_best.csv"), index=False
            )
            pd.DataFrame({"id": dev_cadec["id"].values, "prob": probs_cadec}).to_csv(
                os.path.join(OUTPUT_DIR, "dev_cadec_probs_best.csv"), index=False
            )
            print(f"  *** New best F1={best_f1:.4f} — saved ***")

    # ── Final evaluation with best model ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Best model F1={best_f1:.4f} (seed={SEED})")
    print(f"Thresholds: {json.dumps({k: round(v, 2) for k, v in best_thresholds.items()})}")
    print("Done!")


if __name__ == "__main__":
    main()
