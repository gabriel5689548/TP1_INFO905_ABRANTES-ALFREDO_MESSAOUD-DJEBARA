# === Seuil DAS ===
P_TEACHER_MIN = 0.3  # confiance min du teacher — en dessous, réponse vraiment douteuse

def filter_with_das(data, tokenizer, model, stage_name):
    print(f"\n=== DAS Filtering : {stage_name} ({len(data)} exemples) ===")
    print(f"    Seuil : p_teacher_min > {P_TEACHER_MIN}\n")

    kept = []
    rejected = []

    for i, example in enumerate(data):
        try:
            scores = compute_das_scores(example, tokenizer, model)
            avg_p_teacher = np.mean([s["p_teacher"] for s in scores]) if scores else 0

            example["das_scores"] = scores
            example["avg_p_teacher"] = float(avg_p_teacher)

            if avg_p_teacher >= P_TEACHER_MIN:
                kept.append(example)
                print(f"  [{i+1}/{len(data)}] p_teacher={avg_p_teacher:.4f} → KEEP")
            else:
                rejected.append(example)
                print(f"  [{i+1}/{len(data)}] p_teacher={avg_p_teacher:.4f} → SKIP (teacher incertain)")

        except Exception as e:
            print(f"  [{i+1}/{len(data)}] Erreur: {e}")

    print(f"\n{stage_name} : {len(kept)}/{len(data)} gardés")
    print(f"  Rejetés (teacher incertain) : {len(rejected)}")

    return kept

stage1_filtered = filter_with_das(stage1_data, tokenizer, model, "Stage 1")
stage2_filtered = filter_with_das(stage2_data, tokenizer, model, "Stage 2")

# Sauvegarder
with open("stage1_filtered.json", "w") as f:
    json.dump(stage1_filtered, f, ensure_ascii=False, indent=2)
with open("stage2_filtered.json", "w") as f:
    json.dump(stage2_filtered, f, ensure_ascii=False, indent=2)

print(f"\n=== RÉSUMÉ DAS ===")
print(f"Stage 1 : {len(stage1_filtered)}/{len(stage1_data)} gardés")
print(f"Stage 2 : {len(stage2_filtered)}/{len(stage2_data)} gardés")

# Histogramme
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, name in [(axes[0], stage1_data, "Stage 1"), (axes[1], stage2_data, "Stage 2")]:
    pts = [ex["avg_p_teacher"] for ex in data if "avg_p_teacher" in ex]
    ax.hist(pts, bins=20, edgecolor='black')
    ax.axvline(x=P_TEACHER_MIN, color='red', linestyle='--', label=f'Seuil ({P_TEACHER_MIN})')
    ax.set_title(f"{name} - Confiance Teacher")
    ax.set_xlabel("avg p_teacher")
    ax.set_ylabel("Nombre d'exemples")
    ax.legend()
plt.tight_layout()
plt.savefig("das_distribution.png")
plt.show()
