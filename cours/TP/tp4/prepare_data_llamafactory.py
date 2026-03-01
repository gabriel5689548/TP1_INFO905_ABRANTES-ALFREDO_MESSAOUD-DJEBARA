"""
Convertit les données raw (stage1/stage2) au format ShareGPT pour Llama-Factory.
Applique optionnellement un filtre sur la confiance du teacher.

Usage:
    python prepare_data_llamafactory.py
    python prepare_data_llamafactory.py --threshold 0.6
"""

import argparse
import json
import numpy as np
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful assistant that reasons step by step. "
    "Always structure your reasoning inside <reasoning>...</reasoning> tags "
    "before giving your final answer. Be thorough in your reasoning process."
)


def compute_teacher_confidence(example):
    """Calcule la confiance moyenne du teacher depuis les logprobs."""
    logprobs = example.get("logprobs_data", [])
    if not logprobs:
        return 0.0
    valid = [t["logprob"] for t in logprobs if t.get("logprob") is not None]
    if not valid:
        return 0.0
    return float(np.exp(np.mean(valid)))


def convert_to_sharegpt(examples, with_system_prompt=True):
    """Convertit une liste d'exemples au format ShareGPT pour Llama-Factory."""
    converted = []
    for ex in examples:
        conversations = []
        if with_system_prompt:
            conversations.append({"from": "system", "value": SYSTEM_PROMPT})
        conversations.append({"from": "human", "value": ex["instruction"]})
        conversations.append({"from": "gpt", "value": ex["output"]})
        converted.append({"conversations": conversations})
    return converted


def main():
    parser = argparse.ArgumentParser(description="Prépare les données pour Llama-Factory")
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Seuil de confiance teacher pour filtrer (0 = pas de filtre)"
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Ne pas inclure le system prompt dans les conversations"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Répertoire des fichiers JSON (défaut: même dossier que le script)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir

    for stage_name in ["stage1", "stage2"]:
        raw_path = data_dir / f"{stage_name}_raw.json"
        if not raw_path.exists():
            print(f"ERREUR: {raw_path} introuvable")
            continue

        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n{stage_name}: {len(data)} exemples chargés")

        # Filtrage optionnel par confiance teacher
        if args.threshold > 0:
            filtered = []
            for ex in data:
                conf = compute_teacher_confidence(ex)
                if conf >= args.threshold:
                    filtered.append(ex)
            print(f"  Filtre (p_teacher >= {args.threshold}): {len(filtered)}/{len(data)} gardés")
            data = filtered

        # Conversion au format ShareGPT
        converted = convert_to_sharegpt(data, with_system_prompt=not args.no_system_prompt)

        out_path = data_dir / f"chess_{stage_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        print(f"  Sauvegardé: {out_path} ({len(converted)} exemples)")


if __name__ == "__main__":
    main()
