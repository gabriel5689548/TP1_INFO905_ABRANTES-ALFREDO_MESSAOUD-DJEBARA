# TP4 : Distillation de Modeles de Raisonnement (DASD)

## Distribution-Aligned Sequence Distillation — Application aux Echecs

**Cours** : INFO905
**Etudiants** : ABRANTES-ALFREDO / MESSAOUD-DJEBARA

---

## Objectif

Transferer les capacites de raisonnement d'un modele "enseignant" massif vers un modele "etudiant" compact, en appliquant la methode DASD (Distribution-Aligned Sequence Distillation) au domaine des echecs.

## Stack technique

| Composant | Choix |
|---|---|
| Modele Teacher | `openai/gpt-oss-120b` via API Infomaniak |
| Modele Student | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Framework | Llama-Factory |
| Fine-tuning | LoRA (rank 16, alpha 32) |
| Domaine | Theorie des echecs |

## Structure du projet

```
tp4/
├── INFO905TP4.ipynb            # Notebook principal (toutes les phases)
├── generate_dataset.py         # Script de generation du dataset (API)
├── stage1_raw.json             # 100 exemples, temperature basse (tau=0.3)
├── stage2_raw.json             # 100 exemples, temperature haute (tau=0.9)
├── das_filter_local.py         # Implementation DAS (version locale)
├── das_filter_colab.py         # Implementation DAS (version Colab)
├── prepare_data_llamafactory.py # Conversion au format ShareGPT
├── dataset_info.json           # Registration datasets Llama-Factory
├── stage1_config.yaml          # Config entrainement Stage 1
├── stage2_config.yaml          # Config entrainement Stage 2
├── simple_dasd.py              # Code de demo DAS (fourni)
└── enonce_tp4.md               # Enonce du TP
```

## Deroulement

### Phase 1 : Installation

Installation de Llama-Factory et des dependances sur Google Colab (T4 GPU).

### Phase 2 : Etude du dataset de reference

Exploration du dataset officiel DASD sur HuggingFace (`Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b`) pour comprendre le format attendu : instructions avec reponses structurees utilisant des tags `<reasoning>...</reasoning>`.

### Phase 3 : Generation du dataset via API

- **Domaine choisi** : Theorie des echecs (ouvertures, strategies, milieu de jeu, finales, tactique)
- **100 instructions** couvrant l'ensemble du domaine
- **Teacher** : `openai/gpt-oss-120b` via l'API Infomaniak avec logprobs actives
- **Stage 1** : generation a basse temperature (tau = 0.3) → reponses stables et precises
- **Stage 2** : generation a haute temperature (tau = 0.9) → reponses diversifiees

Chaque reponse est sauvegardee avec ses logprobs pour permettre le filtrage DAS.

### Phase 4 : Divergence-Aware Sampling (DAS)

#### Principe du DAS

Le DAS analyse la divergence phrase par phrase entre le Teacher et le Student :
- **Teacher Sentence** (P_teacher >> P_student) : le Teacher sait, l'etudiant ignore → forte valeur pedagogique
- **Shared Sentence** (P_teacher ~= P_student) : connaissance partagee → neutre
- **Student Sentence** (P_student > P_teacher) : l'etudiant est trop confiant → bruit a rejeter

On conserve une reponse si elle contient une densite suffisante de Teacher Sentences.

#### Adaptation au domaine des echecs

En appliquant le DAS standard, nous avons constate que **100% des exemples etaient conserves**. Cela s'explique par le fait que le modele etudiant (Qwen3-4B) ne connait quasiment rien aux echecs : la divergence Teacher >> Student est **systematiquement positive** pour toutes les phrases, rendant le filtre inefficace (il ne discrimine plus rien).

**Notre solution** : plutot que de comparer Teacher vs Student, nous avons filtre sur la **confiance du Teacher seul** (moyenne geometrique de ses probabilites par phrase). L'intuition est la suivante :
- Si le Teacher est **confiant** sur sa reponse (p_teacher eleve) → la donnee est probablement de bonne qualite → **garder**
- Si le Teacher est **incertain** (p_teacher bas) → la reponse risque de contenir des hallucinations ou des erreurs → **rejeter**

Avec un seuil `P_TEACHER_MIN = 0.3`, cette approche nous a permis d'effectuer un filtrage pertinent adapte a notre domaine specialise. En pratique, la confiance du Teacher etant elevee sur les echecs (0.72-0.87 pour le Stage 1, 0.55-0.79 pour le Stage 2), la majorite des exemples ont ete conserves, ce qui est coherent : le Teacher (gpt-oss-120b) maitrise bien le sujet.

### Phase 5 : Configuration et entrainement

Entrainement en 2 stages suivant le principe du **Temperature-Scheduled Learning** :

| Stage | Donnees | Temperature | Learning Rate | Epochs | Objectif |
|---|---|---|---|---|---|
| Stage 1 | `chess_stage1.json` | tau = 0.3 | 2e-4 | 3 | Apprendre les fondamentaux |
| Stage 2 | `chess_stage2.json` | tau = 0.9 | 1e-4 | 2 | Diversifier le raisonnement |

Le Stage 2 charge l'adapter LoRA du Stage 1 (`adapter_name_or_path`) pour continuer l'apprentissage. Le learning rate est reduit pour ne pas "oublier" les acquis du Stage 1.

Configuration LoRA : rank=16, alpha=32, target=all, dropout=0.05, fp16, gradient_checkpointing.

### Phases 7-9 : Evaluation

- Verification des checkpoints et courbes de loss
- Test du modele distille sur des prompts d'echecs non vus
- Evaluation quantitative : taux de reponses avec `<reasoning>`, longueur du raisonnement, comparaison avant/apres distillation

## Comment reproduire

1. Ouvrir `INFO905TP4.ipynb` sur Google Colab (GPU T4)
2. Executer les cellules dans l'ordre
3. Pour la Phase 5 : uploader `stage1_raw.json` et `stage2_raw.json` quand demande
4. L'entrainement complet prend environ 30-45 minutes sur T4

## Reference

*Base sur le papier "Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning" (Alibaba, 2026)*
