import pandas as pd
import argparse
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import json

def debug_paths():
    print("--- CodaBench Debug Information ---")
    paths_to_check = {
        "Input (Gold Standard/Ref)": "/app/input/ref",
        "Input (Submission/Res)": "/app/input/res",
        "Output": "/app/output",
        "Program Dir": "/app/program"
    }

    for name, path in paths_to_check.items():
        if os.path.exists(path):
            print(f"Directory {name} exists: {path}")
            print(f"Contents: {os.listdir(path)}")
        else:
            print(f"WARNING: Directory {name} NOT FOUND at {path}")

    print("\n--- FULL DIRECTORY TREE ---")
    for root, dirs, files in os.walk('.'):
        for f in files:
            print(os.path.join(root, f))
    print("--- END DIRECTORY TREE ---\n", flush=True)
    print("-----------------------------------")

def load_goldstandard(file_path):
    if os.path.isdir(file_path):
        # Filter for .csv files and ignore hidden files starting with '.'
        files = [f for f in os.listdir(file_path) if f.endswith('.csv') and not f.startswith('.')]
        if len(files) >= 1:
            file_path = os.path.join(file_path, files[0])
        else:
            print(f"Error: No CSV found in {file_path}. Found: {os.listdir(file_path)}", file=sys.stderr)
            sys.exit(1)
    df = pd.read_csv(file_path)
    return df.set_index('id')

def load_predictions(file_path):
    if os.path.isdir(file_path):
        # Filter for .csv files and ignore hidden files starting with '.'
        files = [f for f in os.listdir(file_path) if f.endswith('.csv') and not f.startswith('.')]
        if len(files) >= 1:
            file_path = os.path.join(file_path, files[0])
        else:
            print(f"Error: No CSV found in {file_path}. Found: {os.listdir(file_path)}", file=sys.stderr)
            sys.exit(1)
    df = pd.read_csv(file_path, header=0)
    return df.set_index('id')['predicted_label'].to_dict()


def check_errors(goldstandard, predictions):
    """Check for missing predictions, duplicate entries, and unknown labels in the predictions."""
    errors = []
    goldstandard = goldstandard['label'].to_dict()
    missing_ids = set(goldstandard.keys()) - set(predictions.keys())
    if missing_ids:
        errors.append(f"Missing predictions for IDs: {missing_ids}")

    duplicate_ids = [id for id in predictions if list(predictions.keys()).count(id) > 1]
    if duplicate_ids:
        errors.append(f"Duplicate prediction entries for IDs: {set(duplicate_ids)}")

    unknown_labels = [label for label in predictions.values() if label not in {0, 1}]
    if unknown_labels:
        errors.append(f"Unknown labels found in predictions: {set(unknown_labels)}")

    if errors:
        print("Errors found in predictions:")
        for error in errors:
            print(error)
    else:
        print("No errors found in predictions.")



def evaluate(gold_df, predictions, output_folder=None):
    """Calculate scores and save to scores.json for CodaBench leaderboard."""
    goldstandard = gold_df['label'].to_dict()
    languages = gold_df['language'].to_dict()

    all_y_true = []
    all_y_pred = []

    # This dictionary will hold the final scores for the leaderboard
    final_scores = {}

    # 1. Calculate Per-Language F1
    for lang in set(languages.values()):
        lang_ids = [i for i in goldstandard.keys() if languages.get(i) == lang]
        y_true = [goldstandard[i] for i in lang_ids if i in predictions]
        y_pred = [predictions[i] for i in lang_ids if i in predictions]

        if y_true and y_pred:
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            # KEY MATCHING: Ensure these strings match your competition.yaml columns exactly
            final_scores[f"F1-{lang}"] = round(float(f1), 4)

            # ignore cadec labels for calculating the overall scores
            if not lang.endswith("_cadec"):
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)

    # 2. Calculate Global F1
    if all_y_true and all_y_pred:
        _, _, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary')
        final_scores["F1"] = round(float(f1), 4)

    # 3. Output results
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        # Save scores.json (Preferred by CodaBench)
        score_file = os.path.join(output_folder, "scores.json")
        with open(score_file, "w") as f:
            json.dump(final_scores, f)

        # Legacy scores.txt
        with open(os.path.join(output_folder, "scores.txt"), "w") as f:
            for k, v in final_scores.items():
                f.write(f"{k}: {v}\n")

    # Print to stdout (Scoring Log on CodaBench)
    print("--- FINAL EVALUATION SCORES ---")
    print(json.dumps(final_scores, indent=4))

if __name__ == "__main__":

    debug_paths()

    parser = argparse.ArgumentParser(description="Evaluate predictions against gold standard.")
    parser.add_argument("--goldstandard_file", type=str, help="Path to the goldstandard CSV file")
    parser.add_argument("--predictions_file", type=str, help="Path to the predictions CSV file")
    parser.add_argument("--output_folder", type=str, default=None, help="Optional folder to save evaluation results")
    args = parser.parse_args()


    gold_df = load_goldstandard(args.goldstandard_file)
    predictions = load_predictions(args.predictions_file)

    check_errors(gold_df, predictions)
    evaluate(gold_df, predictions, args.output_folder)