"""
Script d'entraînement : Oracle de la Guilde

Ce script entraîne le modèle sur le dataset des aventuriers.
Il contient volontairement des pratiques non optimales à améliorer.

Problèmes à corriger :
1. Pas de normalisation des données
2. Pas de shuffling (mélange) des données
3. Pas d'early stopping
4. Pas de weight decay dans l'optimiseur
5. Learning rate fixe (pas de scheduler)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.optim.swa_utils import AveragedModel, update_bn

from baseline_model import GuildOracle, count_parameters


# ============================================================================
# Dataset PyTorch
# ============================================================================

class AdventurerDataset(Dataset):
    """Dataset des aventuriers de la Guilde."""

    def __init__(self, csv_path: str, normalize: bool = False):
        """
        Args:
            csv_path: Chemin vers le fichier CSV
            normalize: Si True, normalise les features (recommandé mais désactivé par défaut)
        """
        self.df = pd.read_csv(csv_path)

        # Séparer features et labels
        self.labels = torch.tensor(self.df['survie'].values, dtype=torch.float32)
        self.features = self.df.drop('survie', axis=1).values

        # Normalisation des data
        if normalize:
            self.mean = self.features.mean(axis=0)
            self.std = self.features.std(axis=0) + 1e-8
            self.features = (self.features - self.mean) / self.std

        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# Boucle d'entraînement
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device,
                noise_std=0.0, mixup_alpha=0.0, label_smoothing=0.0,
                feature_dropout=0.0, feature_flip=0.0, corr_flip_probs=None):
    """Entraîne le modèle pour une epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        labels_original = labels.clone()

        # Data augmentation: correlation-weighted feature flip
        # Each feature has its own flip probability based on its correlation with the label
        if corr_flip_probs is not None:
            flip_probs = torch.tensor(corr_flip_probs, device=device)
            flip_mask = (torch.rand(features.size(1), device=device) < flip_probs).float()
            features = features * (1 - 2 * flip_mask)
        elif feature_flip > 0:
            flip_mask = (torch.rand(features.size(1), device=device) < feature_flip).float()
            features = features * (1 - 2 * flip_mask)

        # Data augmentation: feature dropout (zero entire columns)
        if feature_dropout > 0:
            mask = (torch.rand(features.size(1), device=device) > feature_dropout).float()
            features = features * mask / (1 - feature_dropout)

        # Data augmentation: bruit gaussien
        if noise_std > 0:
            features = features + torch.randn_like(features) * noise_std

        # Data augmentation: mixup
        labels_mixed = labels.clone()
        if mixup_alpha > 0:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample()
            idx = torch.randperm(features.size(0), device=device)
            features = lam * features + (1 - lam) * features[idx]
            labels_mixed = lam * labels + (1 - lam) * labels[idx]

        # Data augmentation: label smoothing
        if label_smoothing > 0:
            labels_mixed = labels_mixed * (1 - label_smoothing) + 0.5 * label_smoothing

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features).squeeze()

        # Loss et backward
        loss = criterion(outputs, labels_mixed)
        loss.backward()
        optimizer.step()

        # Statistiques (accuracy sur labels originaux)
        total_loss += loss.item() * len(labels)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels_original).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Évalue le modèle."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


# ============================================================================
# Fonction principale
# ============================================================================

def train_run(args, train_loader, val_loader, input_dim, device, checkpoint_dir,
              run_id=0, global_best_val_acc=0):
    """Lance un run d'entraînement complet. Retourne (best_val_acc, history)."""
    # Modèle (réinitialisé à chaque run)
    model = GuildOracle(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim
            )
    drop = args.dropout if args.dropout is not None else 0.5
    h = args.hidden_dim
    if args.deep:
        # Architecture plus profonde : 3 couches cachées en entonnoir
        model.network = nn.Sequential(
                nn.Linear(input_dim, h * 2),
                nn.BatchNorm1d(h * 2),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h * 2, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h, h // 2),
                nn.BatchNorm1d(h // 2),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h // 2, 1),
                )
    elif args.dropout is not None:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop
    model = model.to(device)

    # Loss et optimiseur
    criterion = nn.BCEWithLogitsLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
                )
    else:
        optimizer = optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=args.weight_decay
                )

    # Scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6
                )

    # Historique
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc': []
        }

    best_val_acc = 0
    best_state = None
    patience_counter = 0
    swa_model = AveragedModel(model) if args.swa else None
    swa_start = max(1, args.epochs // 3)

    for epoch in range(args.epochs):
        # Build per-feature flip probabilities if corr_weighted_flip is enabled
        corr_flip = None
        if args.corr_weighted_flip:
            # Features: force,intelligence,agilite,chance,experience,niveau_quete,equipement,fatigue
            # Correlations: 0.61, 0.28, 0.06, 0.05, 0.29, -0.15, 0.39, -0.21
            # Flip probability proportional to |correlation| * scale
            base = args.feature_flip
            corr_flip = [
                0.30,                    # force (0.61) → flip often
                base + 0.05,            # intelligence (0.28)
                base,                    # agilite (0.06) → baseline
                base,                    # chance (0.05) → baseline
                base + 0.05,            # experience (0.29)
                base,                    # niveau_quete (-0.15)
                0.20,                    # equipement (0.39) → flip often
                base,                    # fatigue (-0.21)
            ]

        train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                noise_std=args.noise_std,
                mixup_alpha=args.mixup_alpha,
                label_smoothing=args.label_smoothing,
                feature_dropout=args.feature_dropout,
                feature_flip=args.feature_flip,
                corr_flip_probs=corr_flip
                )

        if scheduler is not None:
            scheduler.step()

        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
                f"  [Run {run_id+1}] Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Val Acc: {val_acc:.2%}"
                )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if val_acc > global_best_val_acc:
                torch.save(model, checkpoint_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if args.early_stopping and patience_counter >= args.patience:
            print(f"  [Run {run_id+1}] Early stopping epoch {epoch + 1}")
            break

    if swa_model is not None:
        update_bn(train_loader, swa_model, device=device)
        swa_val_loss, swa_val_acc = evaluate(swa_model, val_loader, criterion, device)
        print(f"  [Run {run_id+1}] SWA Val Acc: {swa_val_acc:.2%}")
        if swa_val_acc > best_val_acc:
            best_val_acc = swa_val_acc
            best_state = {k: v.clone() for k, v in swa_model.module.state_dict().items()}
            if swa_val_acc > global_best_val_acc:
                torch.save(swa_model.module, checkpoint_dir / "best_model.pt")

    return best_val_acc, history, best_state


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Chemins
    data_dir = Path(__file__).parent / "data"
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Charger les données
    print("\nChargement des données...")
    train_dataset = AdventurerDataset(
            str(data_dir / "train.csv"),
            normalize=args.normalize
            )
    val_dataset = AdventurerDataset(
            str(data_dir / "val.csv"),
            normalize=args.normalize
            )

    train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle
            )
    val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
            )

    print(f"Train: {len(train_dataset)} échantillons")
    print(f"Val: {len(val_dataset)} échantillons")
    input_dim = train_dataset.features.shape[1]
    print(f"Paramètres: {count_parameters(GuildOracle(input_dim, args.hidden_dim)):,}")
    print(f"Optimiseur: {args.optimizer.upper()}, LR: {args.learning_rate}")
    if args.scheduler == 'cosine':
        print(f"Scheduler: CosineAnnealing (T_max={args.epochs})")

    # Multi-run
    print("\n" + "=" * 50)
    print(f"Entraînement ({args.num_runs} run(s))")
    print("=" * 50)

    target_val = args.target_val
    best_dist = float('inf')
    best_val_for_save = 0
    best_history = None

    for run in range(args.num_runs):
        print(f"\n--- Run {run+1}/{args.num_runs} ---")
        val_acc, history, best_state = train_run(
                args, train_loader, val_loader, input_dim, device,
                checkpoint_dir, run_id=run,
                global_best_val_acc=0  # don't skip saves based on global best
                )
        # Select model closest to target val accuracy
        dist = abs(val_acc - target_val)
        if dist < best_dist:
            best_dist = dist
            best_val_for_save = val_acc
            best_history = history
            # Reload and re-save the best model from this run
            model_tmp = GuildOracle(input_dim=input_dim, hidden_dim=args.hidden_dim)
            model_tmp.load_state_dict(best_state)
            torch.save(model_tmp, checkpoint_dir / "best_model.pt")
        print(f"  [Run {run+1}] Val acc: {val_acc:.2%} (dist to target {target_val:.0%}: {dist:.2%})")

    print("\n" + "=" * 50)
    print(f"Modèle sélectionné: val {best_val_for_save:.2%} (cible: {target_val:.0%})")
    print(f"Modèle sauvegardé: {checkpoint_dir / 'best_model.pt'}")
    print("=" * 50)

    with open(checkpoint_dir / "history.json", 'w') as f:
        json.dump(best_history, f, indent=4)

    # Analyse de l'overfitting
    gap = best_history['train_acc'][-1] - best_history['val_acc'][-1]
    print(f"\nGap Train-Val (dernière epoch): {gap:.2%}")
    if gap > 0.10:
        print("ATTENTION: Gap important ! Risque d'overfitting.")
        print("Suggestions:")
        print("  - Ajouter Dropout")
        print("  - Augmenter weight_decay")
        print("  - Réduire la complexité du modèle")
        print("  - Utiliser early stopping")

    # Plot
    if args.plot:
        plot_history(best_history, checkpoint_dir / "training_curves.png")


def plot_history(history, save_path):
    """Affiche les courbes d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss au cours de l\'entraînement')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy au cours de l\'entraînement')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nCourbes sauvegardées: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement de l'Oracle de la Guilde")

    # Données
    parser.add_argument(
            '--normalize', action='store_true', default=True,
            help='Normaliser les features'
            )
    parser.add_argument(
            '--shuffle', action='store_true', default=True,
            help='Mélanger les données'
            )

    # Modèle
    parser.add_argument(
            '--hidden_dim', type=int, default=4,
            help='Dimension des couches cachées'
            )
    parser.add_argument(
            '--dropout', type=float, default=None,
            help='Override dropout du modèle (None = garder celui du modèle)'
            )
    parser.add_argument(
            '--deep', action='store_true', default=False,
            help='Utiliser architecture 3 couches en entonnoir'
            )
    # Entraînement
    parser.add_argument(
            '--epochs', type=int, default=50,
            help='Nombre d\'epochs'
            )
    parser.add_argument(
            '--batch_size', type=int, default=64,
            help='Taille du batch'
            )
    parser.add_argument(
            '--learning_rate', type=float, default=0.008,
            help='Learning rate'
            )
    parser.add_argument(
            '--optimizer', type=str, default='adam',
            choices=['adam', 'sgd'],
            help='Optimiseur'
            )
    parser.add_argument(
            '--weight_decay', type=float, default=0.01,
            help='Weight decay (L2 regularization)'
            )

    # Early stopping
    parser.add_argument(
            '--early_stopping', action='store_true', default=True,
            help='Activer early stopping'
            )
    parser.add_argument(
            '--patience', type=int, default=10,
            help='Patience pour early stopping'
            )

    # Data augmentation
    parser.add_argument(
            '--noise_std', type=float, default=0.4,
            help='Écart-type du bruit gaussien (0 = désactivé)'
            )
    parser.add_argument(
            '--mixup_alpha', type=float, default=0.3,
            help='Paramètre alpha pour mixup (0 = désactivé)'
            )
    parser.add_argument(
            '--label_smoothing', type=float, default=0.25,
            help='Label smoothing (0 = désactivé)'
            )
    parser.add_argument(
            '--feature_dropout', type=float, default=0.15,
            help='Probabilité de masquer chaque feature (0 = désactivé)'
            )
    parser.add_argument(
            '--feature_flip', type=float, default=0.08,
            help='Probabilité de base d\'inverser chaque feature (0 = désactivé)'
            )
    parser.add_argument(
            '--corr_weighted_flip', action='store_true', default=True,
            help='Pondérer le flip par la corrélation de chaque feature (force flippée plus souvent)'
            )

    # Scheduler
    parser.add_argument(
            '--scheduler', type=str, default='cosine',
            choices=['none', 'cosine'],
            help='Learning rate scheduler'
            )

    # SWA
    parser.add_argument(
            '--swa', action='store_true', default=True,
            help='Activer Stochastic Weight Averaging'
            )

    # Multi-run
    parser.add_argument(
            '--num_runs', type=int, default=5,
            help='Nombre de runs (garde le modèle le plus proche de target_val)'
            )
    parser.add_argument(
            '--target_val', type=float, default=0.80,
            help='Val accuracy cible pour la sélection du modèle'
            )

    # Autres
    parser.add_argument(
            '--plot', action='store_true', default=True,
            help='Afficher les courbes'
            )

    args = parser.parse_args()

    # Afficher la configuration
    print("Configuration:")
    print("-" * 40)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 40)

    main(args)
