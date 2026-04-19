# 🌍 Script de Correction Silver Standard - Darija Marocain

Script Python complet pour corriger sémantiquement un corpus de traduction Darija marocaine en utilisant l'API OpenAI (GPT-4ou-mini).

## 📋 Caractéristiques

✅ **Traitement par batch** : 20 lignes par appel API  
✅ **Gestion d'erreurs robuste** : Retry avec backoff exponentiel (max 3 tentatives)  
✅ **Checkpoints réguliers** : Sauvegarde tous les 500 lignes  
✅ **Rapports détaillés** : Statistiques par shard et rapport global en Markdown  
✅ **Logging complet** : Suivi des erreurs dans `correction_errors.log`  
✅ **Barre de progression** : tqdm pour suivre l'avancement en temps réel  

## 📦 Installation

### 1. Créer et activer un environnement virtuel

```bash
python -m venv env
# Windows:
env\Scripts\Activate.ps1
# Linux/Mac:
source env/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer la clé API

Copier `.env.example` en `.env` et ajouter votre clé OpenAI :

```bash
cp .env.example .env
# Éditer .env et ajouter votre clé:
# OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxx
```

Obtenir votre clé sur : https://platform.openai.com/api-keys

## 🚀 Utilisation

Lancer le script :

```bash
python correct_silver.py
```

## 📂 Structure des fichiers

```
project/
├── correct_silver.py           # Script principal
├── requirements.txt            # Dépendances
├── .env                        # Variables d'environnement (ne pas commiter)
├── .env.example                # Template .env
├── README.md                   # Ce fichier
│
├── shards/
│   └── silver_9k_shards/       # Fichiers d'entrée
│       ├── silver_shard_1.csv  # ~9k lignes
│       ├── silver_shard_2.csv
│       ├── silver_shard_3.csv
│       ├── silver_shard_4.csv
│       └── silver_shard_5.csv
│
├── corrected_silver/           # Résultats corrigés (généré)
│   ├── corrected_silver_shard_1.csv
│   ├── corrected_silver_shard_2.csv
│   ├── corrected_silver_shard_3.csv
│   ├── corrected_silver_shard_4.csv
│   └── corrected_silver_shard_5.csv
│
├── reports/                    # Rapports Markdown (généré)
│   ├── correction_report_shard_1.md
│   ├── correction_report_shard_2.md
│   ├── correction_report_shard_3.md
│   ├── correction_report_shard_4.md
│   ├── correction_report_shard_5.md
│   └── global_correction_report.md
│
├── checkpoints/                # Checkpoints (généré)
│   ├── shard_1_checkpoint.csv
│   ├── shard_2_checkpoint.csv
│   ├── shard_3_checkpoint.csv
│   ├── shard_4_checkpoint.csv
│   └── shard_5_checkpoint.csv
│
└── correction_errors.log       # Fichier de logging (généré)
```

## 📊 Résultats

Après exécution, vous obtiendrez :

### Fichiers corrigés
- `corrected_silver/corrected_silver_shard_X.csv` : Données corrigées avec status = "VALIDATED"

### Rapports détaillés
- `reports/correction_report_shard_X.md` : Pour chaque shard :
  - Résumé exécutif
  - Statistiques par statut d'entrée (GENERATED, PARTIALLY VALIDATED)
  - Statistiques par classe (A, B, C, D)
  - Types de corrections effectuées
  - Détection d'anomalies (Arabizi, champs vides)
  - Exemples de corrections
  - Erreurs (le cas échéant)

- `reports/global_correction_report.md` : Agrégation de tous les shards :
  - Résumé global
  - Comparaison entre shards
  - Distribution globale par statut et classe
  - Statistiques globales

### Checkpoints
- `checkpoints/shard_X_checkpoint.csv` : Sauvegarde progressive tous les 500 lignes

### Logs d'erreur
- `correction_errors.log` : Suivi détaillé avec timestamps et messages d'erreur

## 🔧 Configuration

### Paramètres modifiables dans le script

```python
BATCH_SIZE = 20                    # Lignes par batch API
CHECKPOINT_INTERVAL = 500          # Lignes entre checkpoints
MAX_RETRIES = 3                    # Tentatives max en cas d'erreur
RETRY_BASE_DELAY = 2               # Délai initial de retry (secondes)
TEMPERATURE = 0.1                  # Déterminisme (0 = plus déterministe)
MAX_TOKENS = 4096                  # Tokens max par réponse
```

### Modèle utilisé
- **Modèle** : `gpt-4o-mini`
- **Raison** : Rapide et économe, excellent rapport qualité/prix pour la correction structurée

## 📝 Règles de correction implémentées

### Pour les lignes PARTIALLY VALIDATED
- ✏️ Corrige **UNIQUEMENT** `modern_standard_arabic` si incohérent
- 🔒 Les champs Darija (arabe et arabizi) sont déjà validés

### Pour les lignes GENERATED
- ✏️ Corrige **3 champs simultanément** :
  - `darija_arabic` : Script arabe pur (zéro Arabizi)
  - `darija_arabizi` : Translittération latine fidèle
  - `modern_standard_arabic` : MSA formel cohérent

### Contraintes strictes
- ❌ darija_arabic : aucun caractère latin, aucun chiffre (2, 3, 7, 9)
- ✓ darija_arabizi : translittération phonétique (3=ع, 7=ح, 9=ق, etc.)
- ✓ modern_standard_arabic : MSA formel

### Champs immuables
- 🔐 data_id, id, classe, english, english_word_count

## 🐛 Dépannage

### Erreur : "Variable d'environnement OPENAI_API_KEY non configurée"
**Solution** : Vérifier que le fichier `.env` existe et contient votre clé API OpenAI

### Erreur : "Fichier non trouvé: shards/silver_9k_shards/silver_shard_1.csv"
**Solution** : Vérifier que les 5 fichiers CSV sont dans le bon répertoire

### Erreur API Anthropic (timeout, limite dépassée)
**Solution** : Le script réessaye automatiquement avec backoff exponentiel. Vérifier le log détaillé dans `correction_errors.log`

### Échec du parsing JSON de la réponse API
**Solution** : Le modèle a probablement généré du texte en plus du JSON. Voir les premiers 200 caractères de la réponse dans le log.

## 📈 Performance estimée

- **Temps par batch** : 10-15s (20 lignes)
- **Taux de traitement** : ~1000 lignes/heure
- **Temps total (5 shards × 9000 l.)** : ~45 heures (estimation)
- **Coût API** : ~5-10$ (avec GPT-4o-mini, très compétitif)

## 📞 Support

Pour toute question ou problème :
1. Vérifier les logs dans `correction_errors.log`
2. Consulter la documentation Anthropic : https://docs.anthropic.com
3. Vérifier que votre clé API est valide et a des crédits disponibles

## 📜 Licence

Usage interne - Projet de traduction Darija marocaine.
