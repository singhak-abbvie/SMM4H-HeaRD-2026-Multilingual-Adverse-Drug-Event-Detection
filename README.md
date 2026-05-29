# Gladiators at #SMM4H–HeaRD 2026: Multilingual Adverse Drug Event Detection

**Multi-Seed XLM-RoBERTa Ensemble with Focal Loss and Per-Language Threshold Optimization**

This repository contains the code for the **Gladiators** team's submission to [SMM4H 2026 Task 1](https://healthlanguageprocessing.org/smm4h-2026/) — binary classification of adverse drug event (ADE) mentions in multilingual social media posts.

**Result:** F1 = 0.6039 (above competition mean 0.5465 and median 0.5798)

## Approach

Our system fine-tunes three **XLM-RoBERTa-large** models with different random seeds using:
- **Focal loss** (α=0.75, γ=2.0) to handle severe class imbalance (2.4–10.1% positive)
- **3× positive oversampling** to boost minority class representation
- **Per-language threshold optimization** via grid search on dev set
- **Multi-seed ensemble** averaging predicted probabilities from 3 models (seeds 42, 123, 456)

The system covers 6 training languages (English, German, French, Japanese, Russian, Chinese) and achieved zero-shot transfer to surprise Farsi at test time.

## Repository Structure

```
├── train_xlmr_large_seed.py      # Training script (XLM-R-large with focal loss)
├── ensemble_seeds.py             # Multi-seed ensemble + threshold optimization on dev
├── predict_test_ensemble.py      # Test set prediction with optimized thresholds
├── scoring_task1.py              # Official evaluation script (binary F1)
├── requirements.txt              # Python dependencies
├── training_datasets/            # Train/dev data (6 languages)
│   ├── train_data_SMM4H_2026_Task_1.csv
│   ├── dev_data_SMM4H_2026_Task_1.csv
│   ├── train_data_cadec_translated.csv
│   └── dev_data_cadec_translated.csv
├── test_dataset/                 # Test data (7 languages incl. surprise Farsi)
│   └── combined_test_data_unlabeled.csv
└── submitted_entry_0.60_F1.png   # Competition result screenshot
```

## Setup

```bash
pip install -r requirements.txt
```

**Hardware used:** NVIDIA RTX 5070 Ti (12.8 GB VRAM), peak usage ~11.8 GB.

## Usage

### 1. Train Models (3 seeds)

```bash
python train_xlmr_large_seed.py --seed 42 --output_dir fine_tuned_models/xlmr_large_ade
python train_xlmr_large_seed.py --seed 123 --output_dir fine_tuned_models/xlmr_large_ade_s123
python train_xlmr_large_seed.py --seed 456 --output_dir fine_tuned_models/xlmr_large_ade_s456
```

Each run trains for 5 epochs with cosine LR schedule, saves the best checkpoint by dev F1. Training time: ~35 minutes per seed.

### 2. Ensemble & Optimize Thresholds

```bash
python ensemble_seeds.py
```

This averages probabilities from all 3 models on the dev set and performs per-language threshold grid search over [0.20, 0.80] to maximize F1. Outputs optimized thresholds to `ensemble_output/thresholds.json`.

### 3. Predict on Test Set

```bash
python predict_test_ensemble.py --test_file test_dataset/combined_test_data_unlabeled.csv --output predictions.csv
```

### 4. Evaluate

```bash
python scoring_task1.py --pred predictions.csv --gold gold_standard.csv
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Model | xlm-roberta-large (560M params) |
| Max sequence length | 256 tokens |
| Learning rate | 1e-5 (cosine decay, 10% warmup) |
| Batch size | 32 effective (8 micro × 4 accumulation) |
| Epochs | 5 |
| Focal loss | α=0.75, γ=2.0 |
| Oversampling | 3× positives |
| Precision | fp16 mixed precision |
| Gradient clipping | 1.0 |

## Results

### Dev Set (Pooled Binary F1)

| Configuration | F1 |
|---------------|-----|
| Best single seed (seed 123) | 0.738 |
| 3-seed ensemble | **0.7505** |

### Official Test Set

| Language | F1 | Competition Mean |
|----------|-----|-----------------|
| English | .726 | .685 |
| German | .651 | .664 |
| French | .705 | .681 |
| Japanese | .606 | .534 |
| Russian | .553 | .533 |
| Chinese | .826 | .804 |
| Farsi (zero-shot) | .435 | .367 |
| **Overall** | **.604** | **.547** |

## Key Findings

1. **Model scale matters for low-resource languages**: XLM-R-large gained +15.7% (de), +9.5% (zh) over base
2. **Multi-seed ensembling**: +1.95% F1 from complementary seed-induced errors
3. **Translation augmentation failed**: Domain mismatch (tweets→forums) hurt French by -8.3%
4. **Zero-shot Farsi worked**: XLM-R's multilingual pretraining + shared pharmaceutical vocabulary enabled reasonable transfer (F1=0.435)

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{singh2026gladiators,
  title={Gladiators at \#SMM4H--HeaRD 2026: Multi-Seed XLM-RoBERTa Ensemble with Focal Loss and Per-Language Threshold Optimization for Multilingual Adverse Drug Event Detection},
  author={Singh, Ankit},
  booktitle={Proceedings of the 11th Social Media Mining for Health Research and Applications Workshop and Shared Tasks (SMM4H 2026)},
  year={2026}
}
```

## Author

**Ankit Singh** — singhankit16@gmail.com
