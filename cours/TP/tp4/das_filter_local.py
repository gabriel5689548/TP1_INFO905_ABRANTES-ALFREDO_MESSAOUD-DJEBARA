"""
DAS (Divergence-Aware Sampling) — Filtrage local des données de distillation.

Adapté pour exécution locale (macOS MPS / CUDA / CPU).
Usage :
    python das_filter_local.py
    python das_filter_local.py --model Qwen/Qwen3-4B --threshold 0.2
    python das_filter_local.py --device cpu          # forcer CPU
"""

import argparse
import json
import re
import ssl
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend non-interactif pour sauvegarder sans affichage
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- NLTK avec contournement SSL macOS ---
try:
    _default_https = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    import nltk
    nltk.download("punkt_tab", quiet=True)
    ssl._create_default_https_context = _default_https  # restaurer
    _nltk_available = True
except Exception:
    _nltk_available = False


def sent_tokenize(text: str) -> list[str]:
    """Découpe le texte en phrases. Utilise NLTK si disponible, sinon regex."""
    if _nltk_available:
        try:
            return nltk.tokenize.sent_tokenize(text)
        except Exception:
            pass
    # Fallback regex : coupe sur . ! ? suivis d'un espace ou fin de texte
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def get_device(requested: str | None = None) -> torch.device:
    """Détecte le meilleur device disponible."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str, device: torch.device):
    """Charge le tokenizer et le modèle sur le device choisi."""
    print(f"Chargement de {model_id} sur {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # float16 pour GPU (CUDA/MPS), float32 pour CPU
    dtype = torch.float32 if device.type == "cpu" else torch.float16

    if device.type == "cuda":
        # Sur CUDA, device_map="auto" gère le placement automatiquement
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Sur MPS ou CPU, charger puis déplacer manuellement
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    print(f"Modèle chargé ({dtype}) !")
    return tokenizer, model


def compute_das_scores(example, tokenizer, model, device, max_length=2048):
    """Calcule les scores DAS (P_teacher vs P_student) phrase par phrase."""
    instruction = example["instruction"]
    teacher_text = example["output"]
    teacher_logprobs = example["logprobs_data"]

    # Formater le prompt comme le modèle étudiant s'y attend
    messages = [{"role": "user", "content": instruction}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_input_str = prompt_str + teacher_text

    inputs = tokenizer(
        full_input_str, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    prompt_tokens_len = len(
        tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    )

    # Forward pass unique pour obtenir les logprobs de l'étudiant
    with torch.no_grad():
        logits = model(**inputs).logits

    shift_logits = logits[0, :-1, :]
    shift_labels = inputs["input_ids"][0, 1:]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits, shift_labels)
    token_logprobs_student = -token_losses.cpu().numpy()

    # Libérer la mémoire GPU
    del logits, shift_logits, token_losses, inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    # Extraire les logprobs de la partie réponse uniquement (ignorer le prompt)
    response_logprobs_student = token_logprobs_student[prompt_tokens_len - 1:]
    response_token_ids = shift_labels[prompt_tokens_len - 1:].cpu().numpy()
    del shift_labels

    # Découper en phrases et aligner les scores teacher/student
    sentences = sent_tokenize(teacher_text)
    results = []
    openai_cursor = 0
    qwen_cursor = 0

    for sent in sentences:
        # Score Teacher (depuis les logprobs OpenAI sauvegardés)
        current_accum = ""
        sent_teacher_logprobs = []
        while openai_cursor < len(teacher_logprobs):
            t_data = teacher_logprobs[openai_cursor]
            sent_teacher_logprobs.append(t_data["logprob"])
            current_accum += t_data["token"]
            openai_cursor += 1
            if len(current_accum) >= len(sent):
                break
        p_teacher = (
            float(np.exp(np.mean(sent_teacher_logprobs)))
            if sent_teacher_logprobs
            else 0.0
        )

        # Score Student (depuis le forward pass Qwen)
        current_accum_qwen = ""
        sent_student_logprobs = []
        while qwen_cursor < len(response_token_ids):
            tid = response_token_ids[qwen_cursor]
            token_str = tokenizer.decode([tid])
            sent_student_logprobs.append(response_logprobs_student[qwen_cursor])
            current_accum_qwen += token_str
            qwen_cursor += 1
            if len(current_accum_qwen) >= len(sent):
                break
        p_student = (
            float(np.exp(np.mean(sent_student_logprobs)))
            if sent_student_logprobs
            else 0.0
        )

        results.append({
            "sentence": sent,
            "p_teacher": p_teacher,
            "p_student": p_student,
            "divergence": p_teacher - p_student,
        })

    return results


def filter_with_das(data, tokenizer, model, device, stage_name, threshold=0.2):
    """Filtre un dataset en ne gardant que les exemples riches en Teacher Sentences."""
    print(f"\n{'=' * 60}")
    print(f"DAS Filtering : {stage_name} ({len(data)} exemples)")
    print(f"{'=' * 60}\n")

    scored_data = []
    all_divergences = []
    all_p_teachers = []
    all_p_students = []

    for i, example in enumerate(data):
        try:
            scores = compute_das_scores(example, tokenizer, model, device)
            teacher_sents = sum(1 for s in scores if s["divergence"] > 0.1)
            total_sents = len(scores)
            teacher_ratio = teacher_sents / total_sents if total_sents > 0 else 0.0

            example["das_scores"] = scores
            example["teacher_ratio"] = teacher_ratio
            scored_data.append(example)

            status = "KEEP" if teacher_ratio > threshold else "SKIP"
            print(
                f"  [{i + 1:3d}/{len(data)}] "
                f"Teacher: {teacher_sents}/{total_sents} ({teacher_ratio:.0%}) "
                f"-> {status}"
            )

            # Log par phrase
            for j, s in enumerate(scores):
                truncated = s["sentence"][:50].replace("\n", " ")
                if len(s["sentence"]) > 50:
                    truncated += "..."
                marker = " <-- TEACHER" if s["divergence"] > 0.1 else ""
                print(
                    f"      phrase {j + 1}: "
                    f"p_teacher={s['p_teacher']:.4f}  "
                    f"p_student={s['p_student']:.4f}  "
                    f"div={s['divergence']:+.4f}{marker}"
                )
                print(f"        \"{truncated}\"")

            # Résumé par exemple
            if scores:
                avg_pt = np.mean([s["p_teacher"] for s in scores])
                avg_ps = np.mean([s["p_student"] for s in scores])
                avg_div = np.mean([s["divergence"] for s in scores])
                print(
                    f"    -> Moyennes : p_teacher={avg_pt:.4f}  "
                    f"p_student={avg_ps:.4f}  divergence={avg_div:+.4f}"
                )
                all_divergences.extend(s["divergence"] for s in scores)
                all_p_teachers.extend(s["p_teacher"] for s in scores)
                all_p_students.extend(s["p_student"] for s in scores)

        except Exception as e:
            print(f"  [{i + 1:3d}/{len(data)}] Erreur: {e}")

    kept = [ex for ex in scored_data if ex["teacher_ratio"] > threshold]
    print(f"\n{stage_name} : {len(kept)}/{len(data)} gardés après DAS")

    # Résumé global enrichi du stage
    if all_divergences:
        print(f"  Stats phrases ({len(all_divergences)} phrases totales) :")
        print(
            f"    p_teacher  : moy={np.mean(all_p_teachers):.4f}  "
            f"std={np.std(all_p_teachers):.4f}"
        )
        print(
            f"    p_student  : moy={np.mean(all_p_students):.4f}  "
            f"std={np.std(all_p_students):.4f}"
        )
        print(
            f"    divergence : moy={np.mean(all_divergences):+.4f}  "
            f"std={np.std(all_divergences):.4f}"
        )

    return scored_data, kept


def plot_distribution(stage1_data, stage2_data, threshold, save_path):
    """Histogramme de la distribution des teacher ratios."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, data, name in [
        (axes[0], stage1_data, "Stage 1"),
        (axes[1], stage2_data, "Stage 2"),
    ]:
        ratios = [ex["teacher_ratio"] for ex in data if "teacher_ratio" in ex]
        if ratios:
            ax.hist(ratios, bins=20, edgecolor="black")
            ax.axvline(
                x=threshold, color="red", linestyle="--",
                label=f"Seuil DAS ({threshold})"
            )
        ax.set_title(f"{name} - Distribution Teacher Ratio")
        ax.set_xlabel("Teacher Sentence Ratio")
        ax.set_ylabel("Nombre d'exemples")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Histogramme sauvegardé : {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DAS filtering local pour distillation DASD"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B",
        help="ID du modèle étudiant HuggingFace (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device : cuda, mps, cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Seuil DAS pour garder un exemple (default: 0.2)",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Longueur max de tokenisation (default: 2048)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Répertoire des fichiers JSON (default: même dossier que le script)",
    )
    args = parser.parse_args()

    # Répertoire de données
    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir

    stage1_path = data_dir / "stage1_raw.json"
    stage2_path = data_dir / "stage2_raw.json"

    for p in [stage1_path, stage2_path]:
        if not p.exists():
            print(f"ERREUR : fichier introuvable : {p}")
            sys.exit(1)

    # Charger les données
    print("Chargement des données...")
    with open(stage1_path, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)
    with open(stage2_path, "r", encoding="utf-8") as f:
        stage2_data = json.load(f)
    print(f"Stage 1 : {len(stage1_data)} exemples")
    print(f"Stage 2 : {len(stage2_data)} exemples")

    # Device et modèle
    device = get_device(args.device)
    tokenizer, model = load_model(args.model, device)

    # Filtrage DAS
    stage1_scored, stage1_filtered = filter_with_das(
        stage1_data, tokenizer, model, device, "Stage 1", args.threshold
    )
    stage2_scored, stage2_filtered = filter_with_das(
        stage2_data, tokenizer, model, device, "Stage 2", args.threshold
    )

    # Sauvegarder les résultats
    out_stage1 = data_dir / "stage1_filtered.json"
    out_stage2 = data_dir / "stage2_filtered.json"

    with open(out_stage1, "w", encoding="utf-8") as f:
        json.dump(stage1_filtered, f, ensure_ascii=False, indent=2)
    with open(out_stage2, "w", encoding="utf-8") as f:
        json.dump(stage2_filtered, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("RÉSUMÉ DAS")
    print(f"{'=' * 60}")
    print(f"Stage 1 : {len(stage1_filtered)}/{len(stage1_data)} gardés")
    print(f"Stage 2 : {len(stage2_filtered)}/{len(stage2_data)} gardés")
    print(f"Sauvegardés : {out_stage1}")
    print(f"             {out_stage2}")

    # Histogramme
    plot_path = data_dir / "das_distribution.png"
    plot_distribution(stage1_scored, stage2_scored, args.threshold, plot_path)


if __name__ == "__main__":
    main()
