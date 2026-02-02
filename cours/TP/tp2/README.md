# TP2 : Du Scalaire au Tenseur - Le Tournoi de la Guilde

## Contexte Narratif

Bienvenue dans la **Guilde des Aventuriers** ! Vous venez d'être recruté comme Oracle de la Guilde. Votre mission : prédire si un aventurier survivra à une quête en analysant ses caractéristiques.

La Guilde possède des archives historiques de milliers de quêtes passées. À vous de construire le modèle de prédiction le plus fiable !

## Objectifs Pédagogiques

1. **Maîtriser PyTorch** : Réécrire un MLP en utilisant des tenseurs
2. **Comprendre les dimensions** : Broadcasting, shapes, batching
3. **Optimisation** : Comparer SGD vs Adam, comprendre les learning rates
4. **Généralisation** : Découvrir l'overfitting et les techniques de régularisation

## Structure du TP

```
tp2/
├── README.md                # Ce fichier
├── baseline_model.py        # Modèle de départ (à améliorer !)
├── train.py                 # Script d'entraînement
├── intro_pytorch.ipynb      # Notebook d'introduction à PyTorch
└── data/                    # Données générées
    ├── train.csv
    └── val.csv
```

## Partie 1 : Introduction à PyTorch

### Étape 1 : Comprendre les tenseurs

Ouvrez `intro_pytorch.ipynb` et suivez les exercices sur :
- Création de tenseurs
- Opérations et broadcasting
- Gradients automatiques (`autograd`)
- MLP
- etc

## Partie 2 : Le Tournoi de Généralisation

### Le Défi

Vous recevez un dataset d'aventuriers avec leurs caractéristiques :

| Feature | Description |
|---------|-------------|
| `force` | Force physique (0-100) |
| `intelligence` | Intelligence (0-100) |
| `agilite` | Agilité (0-100) |
| `chance` | Facteur chance (0-100) |
| `experience` | Années d'expérience |
| `niveau_quete` | Difficulté de la quête (1-10) |
| `equipement` | Qualité de l'équipement (0-100) |
| `fatigue` | Niveau de fatigue (0-100) |

**Label** : `survie` (1 = survit, 0 = échec)

### Les Lois de la Survie (Archives Secrètes de la Guilde)

Les Sages de la Guilde ont étudié des milliers de quêtes et ont découvert les facteurs qui déterminent la survie d'un aventurier. Ces connaissances sont transmises uniquement aux Oracles confirmés...

#### Dans les Terres Connues (données d'entraînement)

```
┌─────────────────────────────────────────────────────────────┐
│           FORMULE DE SURVIE - TERRES CONNUES                │
├─────────────────────────────────────────────────────────────┤
│  Équipement ████████████████████████░░░░  25% - Crucial !   │
│  Force      ████████████████████░░░░░░░░  25% - La puissance│
│  Intelligence ████████████████░░░░░░░░░░  20% - La sagesse  │
│  Expérience ████████████░░░░░░░░░░░░░░░░  15% - Le vécu     │
│  Agilité    ████████░░░░░░░░░░░░░░░░░░░░  10% - L'esquive   │
│  Chance     ████░░░░░░░░░░░░░░░░░░░░░░░░   5% - Le destin   │
├─────────────────────────────────────────────────────────────┤
│  MALUS                                                      │
│  Fatigue    ████████████░░░░░░░░░░░░░░░░ -15% - L'épuisement│
│  Difficulté ██████░░░░░░░░░░░░░░░░░░░░░░  -8% - Le danger   │
└─────────────────────────────────────────────────────────────┘
```

*"Un guerrier bien équipé et puissant domine les quêtes classiques."*
— Grimoire de l'Oracle, Chapitre III

#### Dans les Terres Maudites

<details>
<summary>🔒 parchemin des vieux sage</summary>

Les Terres Maudites obéissent à des lois **inversées**. La magie noire qui imprègne ces lieux change tout...

```
┌─────────────────────────────────────────────────────────────┐
│           FORMULE DE SURVIE - TERRES MAUDITES               │
├─────────────────────────────────────────────────────────────┤
│  Intelligence █████████████████████████████ 30% - CRUCIAL ! │
│  Agilité      ████████████████████░░░░░░░░░ 20% - Vital     │
│  Chance       ████████████████████░░░░░░░░░ 20% - Le destin │
│  Équipement   ████████████░░░░░░░░░░░░░░░░░ 15% - Utile     │
│  Force (<70)  ████████░░░░░░░░░░░░░░░░░░░░░ 10% - Modéré    │
│  Expérience   ████░░░░░░░░░░░░░░░░░░░░░░░░░  5% - Peu utile │
├─────────────────────────────────────────────────────────────┤
│  MALUS                                                      │
│  Fatigue      ████████░░░░░░░░░░░░░░░░░░░░ -10%             │
│  Difficulté   ████████░░░░░░░░░░░░░░░░░░░░ -10%             │
│  ARROGANCE    ████████████░░░░░░░░░░░░░░░░ -15% (Force >70!)│
└─────────────────────────────────────────────────────────────┘
```

**LE PIÈGE DE L'ARROGANCE** : Les guerriers trop confiants en leur force (>70) subissent une pénalité ! Leur arrogance les rend vulnérables aux pièges magiques des Terres Maudites.

*"Dans les Terres Maudites, la ruse vaut mieux que la force brute."*
— Inscription sur une stèle oubliée

**Leçon pédagogique** : Les modèles qui ont mémorisé "force = survie" échoueront. Seuls les modèles régularisés qui ont appris des patterns généraux s'adapteront.

**Attention**: !!!! Dans la terre maudites l'atmosphère normalise les données de test !!!

</details>

### Règles du Tournoi

1. **Complétez** Le model oracle [baseline_model.py](baseline_model.py)
1. **Entraînez** votre modèle a l'aide de `uv run train.py`
1. **Soumettez** Uploader votre meilleur fichier `.pt` dans l'interface web fournit par le maitre du jeu
1. Le classement final sera basé sur un **test secret** !

### Le Twist

Le dataset de test secret contient des aventuriers partis en quête dans les **Terres Maudites**, où les règles sont légèrement différentes...

Ceux qui ont sur-appris les données d'entraînement seront surpris !

**Attention**: !!!! Dans la terre maudites l'atmosphère normalise les données de test !!!

### Conseils

Questions à vous poser :
- Mon modèle est-il trop complexe pour la quantité de données ?
- Est-ce que j'utilise de la régularisation (Dropout, Weight Decay) ?
- Est-ce que je fais de l'early stopping ?
- Mon modèle généralise-t-il ou mémorise-t-il ?

## Commandes Utiles

```bash
# Générer les données
uv run train_oracle.py
```

## Ressources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- Notebook `intro_pytorch.ipynb` pour les bases

---

*Que la chance soit avec vous, jeune Oracle !*

---

## Partie 3 : Les Archives Interdites - Oracle du Donjon (Séquences)

### Le Nouveau Défi

Après avoir maîtrisé la prédiction basée sur les statistiques, la Guilde vous confie une mission plus complexe : analyser les **journaux de donjon** pour prédire la survie des aventuriers.

Cette fois, ce n'est plus un simple tableau de stats, mais une **séquence d'événements** !

> La meilleur équipe sera celle qui arrive a la meilleur accuracy mais avec le model le plus petit possible !

### Structure des Données

```
tp2/
├── data/
│   ├── train_dungeon.csv      # Données d'entraînement (séquences)
│   ├── val_dungeon.csv        # Données de validation
│   └── vocabulary_dungeon.json # Vocabulaire des événements
├── baseline_model.py          # Contient DungeonOracle (à améliorer !)
├── train_dungeon_logs.py      # Script d'entraînement séquences
└── app_leaderboard_dungeon.py # Interface de soumission
```

### Format des Données

Chaque aventurier est représenté par une **séquence d'événements** :

```
Entree -> Rat -> Potion -> Coffre -> Gobelin -> Dragon -> Sortie
```

| Token | Description |
|-------|-------------|
| `Entree` / `Sortie` | Début et fin du donjon |
| `Rat`, `Gobelin`, `Orc`, `Troll`, `Dragon` | Monstres (dégâts croissants) |
| `Potion`, `Feu_de_Camp`, `Fontaine_Sacree` | Soins |
| `Piege_a_Pics`, `Fleches_Empoisonnees`, `Fosse` | Pièges |
| `Coffre`, `Gemmes`, `Or`, `Relique` | Trésors |
| `Amulette_Protection`, `Armure_Ancienne`, `Epee_Legendaire` | Objets spéciaux |

**Label** : `survived` (1 = survit, 0 = mort)

### Les Lois des Donjons (Archives Secrètes)

<details>
<summary>🔒 Parchemin des Archivistes</summary>

#### L'ORDRE COMPTE !

```
┌─────────────────────────────────────────────────────────────┐
│              RÈGLE D'OR DES DONJONS                         │
├─────────────────────────────────────────────────────────────┤
│  ✅ Potion -> Dragon    = SURVIE (soigné avant le combat)   │
│  ❌ Dragon -> Potion    = MORT   (trop tard pour se soigner)│
└─────────────────────────────────────────────────────────────┘
```

#### DÉPENDANCES LONG-TERME

- L'`Amulette_Protection` au **début** protège contre le Boss **final**
- L'`Epee_Legendaire` trouvée tôt facilite **tous** les combats suivants
- La `fatigue` s'accumule : trop de combats sans repos = danger

*"L'ordre des épreuves détermine le destin de l'aventurier."*
— Inscription sur les Archives Interdites

</details>

### Architecture Baseline : DungeonOracle

A vous d'explorer le code pour trouver le meilleur moyen de crée votre model le plus petit mais également le plus performant !

### Commandes d'Entraînement

```bash
# Entraînement baseline (non optimal)
uv run train_dungeon_logs.py
```

### Règles du Tournoi Dungeon

Trouver le meilleur model mais également le plus petit !

### Le Twist

Le dataset de test secret contient des séquences de donjons **plus longues** et avec des **patterns inédits**...

Les modèles qui ont mémorisé les séquences d'entraînement échoueront !

---

*Que les Archives vous guident, jeune Oracle !*
