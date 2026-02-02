# Exercice 4: Implementer la fonction Sigmoid

> Master 2 Informatique - Introduction IA

## Objectif

Implementer la fonction d'activation sigmoid dans `engine.py` et verifier que le gradient est correctement calcule.

## Formules

```
sigmoid(x) = 1 / (1 + e^(-x))

d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
```

## Description

La fonction sigmoid est une fonction d'activation classique qui "ecrase" les valeurs entre 0 et 1.

### Proprietes

| Propriete | Valeur |
|-----------|--------|
| `sigmoid(0)` | 0.5 |
| `sigmoid(x)` quand x -> +infini | 1 |
| `sigmoid(x)` quand x -> -infini | 0 |
| Derivee maximale (en x=0) | 0.25 |

## Votre Mission

Dans le fichier `micrograd/engine.py`, ajoutez la methode `sigmoid()` a la classe `Value`:

## Bonus: Sigmoid vs ReLU

La sigmoid etait tres populaire avant l'avenement du deep learning. Aujourd'hui, ReLU est preferee car:

1. **Gradient vanishing**: sigmoid sature pour |x| > 5, gradient proche de 0
2. **Non-zero centered**: sortie toujours > 0, peut ralentir la convergence
3. **Calcul**: `exp()` est plus couteux que `max(0, x)`

Cependant, sigmoid reste utile pour:
- Couche de sortie en classification binaire (probabilite entre 0 et 1)
- Gates dans LSTM/GRU

### A faire

Tester sigmoid vs ReLU dans votre MLP et comparer les resultats.
