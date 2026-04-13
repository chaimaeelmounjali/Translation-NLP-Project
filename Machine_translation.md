# Cahier des charges du projet R&D : Traduction Automatique Darija (MT)
**Constitution, Annotation Semi-Automatique et Modélisation**

## 1. Contexte & Objectifs — Du Silver au Gold

La **traduction automatique (Machine Translation – MT)** vers les langues dites "low-resource" comme la **Darija marocaine** est un défi majeur du NLP moderne. Ce projet s'inscrit dans une démarche pédagogique et scientifique visant à construire un corpus parallèle de haute qualité tout en contournant le problème du "Cold Start" (démarrage à froid).

Plutôt que de collecter des données à partir de zéro, vous recevez un **"Silver Standard"** : un pack de **10 000 lignes pré-alignées** par des modèles d'IA (Atlas/Gemini) ou issues de bases existantes.

**Votre mission R&D :**
Transformer ce corpus brut en un **"Gold Standard"** via :
1.  **Audit & Nettoyage (DQA)** : Utiliser l'analyse de données pour repérer les erreurs d'alignement.
2.  **Validation Humaine** : Corriger les prédictions imparfaites (Silver) et traduire manuellement le set de test (Gold).
3.  **Modélisation** : Créer un modèle de traduction **AraT5v2** performant.

---

## 2. Le Jeu de Données : Structure & Défi

Chaque groupe reçoit un **pack de données** composé de deux fichiers principaux situés dans le dossier `shards/`.

### 2.1 Composition du Shard
| Composant | Volume | Fichier | Rôle |
| :--- | :--- | :--- | :--- |
| **Shard Principal** | ~9 000 | `unified_shard_X.csv` | **Training Set**. Données pré-remplies (Silver) à auditer. |
| **Gold Standard** | ~1 000 | `gold_shard_X.csv` | **Holdout Test**. Données vierges (ou brutes) à traduire manuellement (**Zéro IA**). |

> **Note**: Les fichiers contiennent des colonnes pour Darija (Arabe & Arabizi), Anglais et MSA (Modern Standard Arabic).

### 2.2 Logique de Validation (Les 3 Statuts)
Pour maximiser l'efficacité, la stratégie de validation dépend du statut de la donnée :

1.  **`PARTIALLY VALIDATED`** (Source: Doda)
    *   **État** : Darija ↔ English validés. MSA généré par IA.
    *   **Action** : **Valider uniquement le MSA**. Vérifiez grammaire et fidélité.

2.  **`GENERATED`** (Source: Wikipedia/Web)
    *   **État** : Tout est généré ou aligné automatiquement (Risque élevé d'hallucinations).
    *   **Action** : **Revue Complète**. Vérifiez tout. Convertissez impérativement l'**Arabizi** en script Arabe si nécessaire.

3.  **`TO_TRANSLATE`** (Source: Gold Set)
    *   **État** : Darija présent. Cibles vides ou à ignorer.
    *   **Action** : **Traduction Manuelle (0 IA)**. C'est votre pure vérité terrain pour l'évaluation.

### 2.3 Structure du Fichier CSV Final
Vous devez **impérativement** respecter la structure suivante pour le fichier CSV final. Le jeu de données doit être riche et inclure la translittération (Arabizi).

| Colonne | Description | **Exemple 1 (Prêt)** | **Exemple 2 (En Cours)** |
| :--- | :--- | :--- | :--- |
| `data_id` | Tech ID | `data1` | `data5001` |
| `id` | Unique ID | `DATA_001` | `DATA_5001` |
| `classe` | Plage de mots | `A` | `A` |
| `darija_arabic` | Darija (Lettres Arabes) | *شكرا بزاف على هاد الهدية الزوينة* | *البارح كنت باغي نمشي للسوق ساعا طاحت شتا بزاف* |
| `darija_arabizi` | Darija (Latin) | *Choukran bzaf 3la had l-hadiya zwina* | *LBareh kont baghi nemchi l souk sa3a ta7et chtat bzaf* |
| `english` | Anglais | *Thank you so much for this nice gift* | *(Vide)* |
| `modern_standard_arabic` | Arabe Standard | *شكرا جزيلا على هذه الهدية الجميلة* | *(Vide)* |
| `status` | État | <span style="color:green">**VALIDATED**</span> | <span style="color:red">**TO_TRANSLATE**</span> |
| `dataset_type` | Provenance | `UNIFIED` | `GOLD` |

> **Cycle de Vie & Validation** :  
> Le processus de validation se déroule en 3 étapes :
> 1. <span style="color:red">**TO_TRANSLATE**</span> : Case traductions vide. L'étudiant doit générer une pré-traduction via IA (Google Colab).
> 2. <span style="color:orange">**GENERATED**</span> ou **`PARTIALLY VALIDATED`** : Une traduction IA existe mais n'a pas encore été relue. C'est l'état d'entrée pour **Label Studio**.
> 3. <span style="color:green">**VALIDATED**</span> : L'étudiant a **accepté ou corrigé** la traduction. C'est le seul état accepté dans le livrable final.

---

## 3. Phase 1 : Exploration & Analyse (EDA)

Avant toute annotation, vous devez comprendre la qualité de votre shard. Utilisez le notebook **`EDA.ipynb`** fourni.

**Objectifs de l'EDA :**
*   **Détection d'Anomalies** : Identifier les lignes où l'anglais contient de l'arabe, ou le Darija contient des caractères latins (Arabizi non converti).
*   **Balance du Corpus** : Vérifier la distribution des 4 classes de longueur (A, B, C, D).
*   **Qualité Initiale** : Estimer le taux d'erreur du "Silver Standard" pour calibrer votre effort.

> **Instruction** : Ne modifiez pas le fichier source CSV directement. Notez les IDs problématiques pour les traiter en priorité dans Label Studio.

---

## 4. Phase 2 : Protocole d'Annotation (Label Studio)

L'objectif est d'atteindre 100% de lignes au statut **`VALIDATED`**.

### 4.1 Configuration
1.  Lancer Label Studio : `label-studio start`.
2.  Importer le XML de configuration fourni (`Labeling_Interface.xml`).
3.  Importer d'abord le **Gold Set** (`shards/1k_gold_shards/`) puis le **Shard Principal** (`shards/unified_shards/`).

### 4.2 Guide de Correction
*   **Fidélité Sémantique** : La traduction doit véhiculer le même sens, ton et niveau de langage que la source.
*   **Normalisation du Script** :
    *   Tout le Darija doit finir en **script Arabe** dans la colonne `darija_arabic`.
    *   Si la source est en Arabizi (ex: *kech*), traduisez-la en Arabe (*مراكش*) dans la correction.
*   **Code-Switching** : Gérez les mélanges de langues de manière naturelle. Ne forcez pas une "pureté" linguistique artificielle si le Darija original utilise des termes français/espagnols intégrés.

---

## 5. Pipeline d'Annotation

### Phase 1 : Le Gold Standard (100% Manuel & Validé)
Avant toute automatisation, nous devons définir la vérité terrain.
- **Sélection** : 1 000 phrases issues de votre Gold Set.
- **Action** : Traduction **100% manuelle** par les étudiants (**Zéro IA**) pour constituer la vérité terrain.
- **<span style="color:red">Livrable</span>** : Création du **Guide de Traduction** (Convention sur l'orthographe des noms propres, temps verbaux, gestion de l'Arabizi).
- **Usage** : Ce jeu de données sera verrouillé et servira uniquement au **Test Set final**.

### Phase 2 : Pré-traduction (Semi-Automatique)
Pour les lignes manquantes ou les validations difficiles, utilisation d'assistants IA pour le "First Draft".

**Outils Recommandés :**
Vous pouvez utiliser les fonctionnalités IA gratuites de Google Colab (Gemini).
- *Lien* : [Getting started with Google Colab AI](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/Getting_started_with_google_colab_ai.ipynb#scrollTo=R7taibpc7x2l)

**Exemple de génération de batch :**
```python
# @title Assistant de Traduction & Validation (Google Colab AI)
# Nécessite un notebook Google Colab actif
from google.colab import ai

# Votre phrase Darija difficile
darija_text = "مؤسسة محمد السادس للأمن الوطني دارت بروكَرام فيه منح دراسية لولاد وأيتام موظفي البوليس"

# Prompt pour Gemini (intégré à Colab)
prompt = f"""
Agis comme un traducteur expert marocain. 
1. Traduis cette phrase Darija en Anglais (Direct & Meaningful).
2. Traduis cette phrase en Arabe Standard (MSA) formel.
Phrase Darija : '{darija_text}'
"""

try:
    response = ai.generate_text(prompt)
    print(f"--- Suggestion Gemini ---\n{response}")
except:
    print("Utilisez Google Colab pour cette fonction.")

# Résultat attendu :
# The Mohammed VI Foundation for National Security organized a program which includes scholarships for the children and orphans of police staff.
```

- **Prompt Engineering** : "Tu es un traducteur expert dialectal. Traduis ce texte Darija en Anglais/Arabe en restant fidèle au registre informel."

### Phase 3 : Validation Humaine (Label Studio)
L'étape critique de "Human-in-the-loop".
- **Import** : Charger le CSV contenant `darija_arabic` et les drafts.
- **Tâche Annotateur** :
    - Si le draft est parfait → **Accepter**.
    - Si le draft contient une erreur → **Corriger** (Post-édition).
    - Si le draft est hors-sujet → **Rejeter**.

---

## 6. Phase 3 : Modélisation (AraT5v2)

Une fois le corpus validé, vous passerez à l'entraînement d'un modèle neuronal.

### 5.1 Architecture
Vous utiliserez **AraT5v2** (ou des variantes équivalentes comme DarijaT5 si disponible), un modèle Transformer Sequence-to-Sequence pré-entraîné sur l'arabe.

### 5.2 Entraînement
*   **Plateforme** : Google Colab (recommandé pour l'accès GPU).
*   **Split** : Train (Shard validé) / Dev (10% Shard) / Test (Gold Set).
*   **Hyperparamètres** :
    *   Learning Rate : 3e-5
    *   Batch Size : 16/32
    *   Epochs : 5 à 10 (Early Stopping sur la loss de validation).

### 5.3 Tracking Expérimental
L'utilisation de **MLflow** ou **Weights & Biases** est requise pour tracer vos expériences (Loss curves, BLEU scores par epoch).

---

## 6. Phase 4 : Évaluation & Benchmarking

Vous devez prouver la qualité de votre modèle sur le **Gold Standard**.

### 6.1 Métriques Automatiques
*   **BLEU** : Score de précision n-gram (Standard de l'industrie).
*   **chrF++** : Score basé sur les caractères (Meilleur pour les langues morphologiquement riches comme le Darija).
*   **BERTScore** : Similarité sémantique (embbedings).

### 6.2 Analyse Qualitative
Ne vous contentez pas de chiffres. Analysez :
*   Les erreurs sur les phrases longues vs courtes.
*   La gestion des entités nommées et du code-switching.

---

## 7. Livrables Attendus

Pour un projet de niveau R&D, vous devez fournir :

### 7.1 Code & Tracking
1.  **Code Source Documenté** : Scripts de pré-traitement, Notebooks de Fine-tuning et Notebooks des Baselines (si utilisées).
2.  **Dashboard MLflow** : Preuve des expérimentations (loss curves, scores BLEU/chrF++).
3.  **`requirements.txt`** et **README.md**.

### 7.2 Données & Annotation
4.  **Gold Standard Final** : Votre `gold_shardX.csv` entièrement traduit manuellement (Zéro IA).
5.  **Shard Principal Validé** : Le fichier `unified_shard_X.csv` avec 100% de statuts `VALIDATED`.
6.  **Export Label Studio** : Fichier JSON brut de vos annotations.

### 7.3 Rapport Technique (PDF)
7.  **Rapport Académique** : Un document structuré détaillant la méthodologie, l'EDA, le protocole d'annotation et le benchmarking des modèles.

---

## 8. Extensions & Bonus (Pour des points supplémentaires)

### Bonus 1 — Traduction Bidirectionnelle Complète
- Traduction bidirectionnelle complète **Darija ↔ Anglais ↔ MSA**.

### Bonus 2 — Étude du Code-Switching
- Étude de l'impact du code-switching (Darija/Français/Anglais) sur la qualité de la traduction.

### Bonus 3 — Comparaison de Modèles
- Comparaison LLM (Gemini/GPT) vs Modèles spécialisés (AraT5v2) entraînés localement.

### Bonus 4 — Open Science
- Publication du corpus (open science).

---

## 9. Critères d'Évaluation

Le projet sera évalué selon la pondération suivante :

| Critère | Pondération |
|------|-------------|
| Qualité du corpus et annotation | 50 % |
| Pipeline NLP / traduction | 15 % |
| Modélisation et évaluation | 10 % |
| Tracking et reproductibilité | 10 % |
| Analyse critique | 15 % |

Concluding note: utilisez le corpus Silver pour expérimenter le Transfer Learning, la Distillation et une évaluation rigoureuse tout en maintenant une validation humaine forte.

