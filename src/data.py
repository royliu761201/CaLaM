
import json
import os
import random
from typing import List, Dict, Optional

class DatasetLoader:
    """Base class for CaLaM evaluation datasets."""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.samples = []

    def load(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class RealToxicityPromptsLoader(DatasetLoader):
    """Loader for RealToxicityPrompts."""
    def load(self, split: str = "train", challenging_only: bool = False):
        if not os.path.exists(self.data_path):
            # AB-038: DUMMY MODE PROHIBITED
            raise FileNotFoundError(
                f"[AB-038] CRITICAL: Dataset file not found at {self.data_path}. "
                "Manual inspection required. Do NOT proceed with dummy data."
            )

        with open(self.data_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if challenging_only:
                        tox = data.get("prompt", {}).get("toxicity")
                        if tox is None or tox < 0.5:
                            continue
                    self.samples.append(data)
                except:
                    continue
        print(f"Loaded {len(self.samples)} RTP samples (challenging_only={challenging_only}).")

class SafetyBenchLoader(DatasetLoader):
    """Loader for SafetyBench."""
    def load(self, split: str = "test"):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[FAIL-FAST] SafetyBenchLoader: data not found at {self.data_path}. "
                "Mount dataset before running."
            )
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list):
                            self.samples.extend(v)
                        else:
                            self.samples.append(v)
                elif isinstance(data, list):
                    self.samples = data
            print(f"Loaded {len(self.samples)} SafetyBench samples.")
        except Exception as e:
            raise RuntimeError(f"[FAIL-FAST] SafetyBench load failed: {e}") from e

class TruthfulQALoader(DatasetLoader):
    """Loader for TruthfulQA — converts to MC format (ABCD).

    CSV columns: Type, Category, Question, Best Answer, Best Incorrect Answer,
                 Correct Answers, Incorrect Answers, Source

    Strategy: Use 'Best Answer' as correct choice + up to 3 from 'Incorrect Answers'
    to form A/B/C/D options. Shuffle, record correct label.
    """
    def load(self, split: str = "validation"):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[FAIL-FAST] TruthfulQALoader: data not found at {self.data_path}."
            )
        import csv
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = row["Question"]
                    correct = row["Best Answer"].strip()

                    # Parse incorrect answers (semicolon-separated)
                    incorrect_raw = row.get("Incorrect Answers", "")
                    incorrects = [a.strip() for a in incorrect_raw.split(";") if a.strip()]

                    # Take up to 3 incorrect answers
                    if len(incorrects) < 3:
                        # Pad with "Best Incorrect Answer" if not already present
                        best_incorrect = row.get("Best Incorrect Answer", "").strip()
                        if best_incorrect and best_incorrect not in incorrects:
                            incorrects.insert(0, best_incorrect)
                    incorrects = incorrects[:3]

                    if len(incorrects) < 1:
                        continue  # Need at least 1 wrong answer

                    # Pad to 3 if needed (duplicate last)
                    while len(incorrects) < 3:
                        incorrects.append(incorrects[-1] + " (variant)")

                    # Build MC: correct is always first, then shuffle deterministically
                    options = [correct] + incorrects  # [correct, wrong1, wrong2, wrong3]

                    # Deterministic shuffle based on question hash
                    import hashlib
                    seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
                    rng = random.Random(seed)
                    indices = list(range(4))
                    rng.shuffle(indices)
                    shuffled = [options[i] for i in indices]
                    correct_idx = indices.index(0)  # Where did the correct answer end up?
                    label = ["A", "B", "C", "D"][correct_idx]

                    prompt = f"{question}\nA. {shuffled[0]}\nB. {shuffled[1]}\nC. {shuffled[2]}\nD. {shuffled[3]}\nAnswer:"

                    self.samples.append({
                        "prompt": prompt,
                        "answer": label
                    })
            print(f"Loaded {len(self.samples)} TruthfulQA samples (MC format).")
        except Exception as e:
            raise RuntimeError(f"[FAIL-FAST] TruthfulQA load failed: {e}") from e

class JailbreakBenchLoader(DatasetLoader):
    """Loader for JailbreakBench (Adversarial Prompts)."""
    def load(self, split: str = "test"):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[FAIL-FAST] JailbreakBenchLoader: data not found at {self.data_path}."
            )
        import csv
        try:
            if os.path.isdir(self.data_path):
                csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
                if not csv_files:
                    raise FileNotFoundError(f"[FAIL-FAST] No CSV files in JailbreakBench dir: {self.data_path}")
                file_path = os.path.join(self.data_path, csv_files[0])
            else:
                file_path = self.data_path

            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "goal" in row: row["prompt"] = row["goal"]
                    self.samples.append(row)
            print(f"Loaded {len(self.samples)} JailbreakBench samples.")
        except Exception as e:
            raise RuntimeError(f"[FAIL-FAST] JailbreakBench load failed: {e}") from e

class MMLULoader(DatasetLoader):
    """Loader for MMLU from offline CSV (pre-downloaded via Kaggle Worker).

    CSV format: question, subject, choices (string repr of list), answer (int 0-3).
    """
    def load(self, split: str = "test"):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[FAIL-FAST] MMLULoader: data not found at {self.data_path}. "
                f"Use Kaggle Worker to download and SCP to GPU first."
            )
        import csv, ast
        try:
            print(f"Loading MMLU from offline CSV: {self.data_path}")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    question = row["question"]
                    # choices is stored as string repr of list, e.g. "['0' '4' '2' '6']"
                    raw_choices = row["choices"]
                    try:
                        choices = ast.literal_eval(raw_choices.replace("' '", "', '"))
                    except Exception:
                        choices = [c.strip().strip("'\"") for c in raw_choices.strip("[]").split("' '")]

                    if len(choices) < 4:
                        continue  # Skip malformed rows

                    prompt = f"{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

                    tgt_idx = int(row["answer"])
                    label = ["A", "B", "C", "D"][tgt_idx]

                    self.samples.append({
                        "prompt": prompt,
                        "answer": label,
                        "subject": row["subject"]
                    })

            # Shuffle deterministically
            import random
            random.seed(42)
            random.shuffle(self.samples)
            print(f"Loaded {len(self.samples)} MMLU samples (offline).")
        except Exception as e:
            raise RuntimeError(f"[FAIL-FAST] MMLU load failed: {e}") from e

class XSTestLoader(DatasetLoader):
    """Loader for XSTest (Exaggerated Safety Test)."""
    def load(self, split: str = "test"):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[FAIL-FAST] XSTestLoader: data not found at {self.data_path}."
            )
        import csv
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append({
                        "prompt": row["prompt"],
                        "type": row["type"]
                    })
            print(f"Loaded {len(self.samples)} XSTest samples.")
        except Exception as e:
            raise RuntimeError(f"[FAIL-FAST] XSTest load failed: {e}") from e
