# 🌐 Darija Machine Translation & Semantic Quality Control Pipeline (Group 3)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg?logo=openai)](https://openai.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203%20Batch-orange.svg)](https://groq.com/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-QC%20Filtering-blue.svg)](https://www.sbert.net/)
[![Report](https://img.shields.io/badge/Report-LaTeX%20%2F%20Beamer-red.svg)](Rapport_QC_Cleaning_MT.pdf)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Bilingual README: [Français](#-version-française) | [English](#-english-version)*

---

## 🇫🇷 Version Française

### 🎯 Objectif
Ce projet constitue le sous-système d'ingénierie et de contrôle qualité sémantique pour la constitution d'un corpus bimodal de traduction automatique de référence (**Anglais ↔ Arabe Standard Moderne (MSA) ↔ Darija Marocaine** en graphie arabe et en Arabizi). L'objectif est d'éliminer les erreurs de traduction automatique (hallucinations, désalignements syntaxiques et contresens) sur le **Silver Standard (Shard 3)** à l'aide d'un pipeline automatisé de correction par LLM (OpenAI / Groq) et de validation sémantique hybride par *sentence embeddings*.

### 🛠️ Stack Technologique
- **Langage & Environnement** : Python 3.10+, Jupyter Notebooks (`EDA_gold_shard_3_comparatif.ipynb`, `EDA_silver_shard_3_comparatif.ipynb`, `Hybrid_Semantic_QC_Visualization.ipynb`).
- **LLM APIs & Batching** : OpenAI API (`gpt-4o-mini`), Groq API (Llama 3 70B), requêtes asynchrones, reprise sur erreur avec backoff exponentiel et gestion de checkpoints (`.jsonl`).
- **Contrôle Qualité & NLP** : Sentence-Transformers (`paraphrase-multilingual-mpnet-base-v2`, similarité cosinus sémantique), Label Studio (`Labeling_Interface.xml`).
- **Analyse de Données & Reporting** : Pandas, NumPy, Matplotlib, Seaborn, LaTeX / Beamer (`Rapport_QC_Cleaning_MT.pdf`, `Presentation_Projet_Traduction.pdf`).

### 👩‍💻 Mon Rôle & Contributions
- **Pipeline de Correction Automatisée (`correct_silver.py`, `correct_all_4_mt_fields.py`)** :
  - Conception de l'architecture de traitement par batch (20 lignes/appel) avec gestion de checkpoints toutes les 500 lignes.
  - Implémentation du système de reprise automatique en cas d'interruption réseau ou de rate limit API.
- **Contrôle Qualité Sémantique Hybride (`hybrid_semantic_consistency.py`)** :
  - Mise en place d'un algorithme calculant la similarité vectorielle entre phrases sources et traduites.
  - Filtrage automatique des paires de traduction dont la similarité sémantique est inférieure aux seuils de confiance.
- **Analyse Exploratoire Comparative (EDA)** :
  - Conception des notebooks comparant la distribution de longueur des tokens, la richesse lexicale et les taux d'erreur entre le Gold Shard 3 annoté manuellement et le Silver Shard 3 corrigé.
- **Rédaction Académique & Présentation** :
  - Rédaction intégrale du rapport technique en LaTeX (`Rapport_QC_Cleaning_MT.pdf`) et des diapositives de soutenance Beamer (`Presentation_Projet_Traduction.pdf`).

### 📊 Résultats & Métriques Clés
- **Volume traité et purifié** : Plus de 7 000 segments de phrases corrigés et harmonisés avec un taux de rétention de sens > 95%.
- **Réduction des faux alignements** : Détection et correction automatisée de plus de 80% des artefacts de traduction littérale grâce au filtrage par embeddings.
- **Livrables complets** : Rapport scientifique PDF et diapositives de présentation compilés et intégrés au dépôt.

---

## 🇬🇧 English Version

### 🎯 Objective
This repository hosts the data engineering, semantic quality control (QC), and automated correction pipeline for a tri-lingual parallel translation corpus bridging **English, Modern Standard Arabic (MSA), and Moroccan Darija** (Arabic script and Latin Arabizi). The mission is to systematically detect and resolve machine translation deficiencies (hallucinations, register mismatches, dropped context) across the **Silver Standard (Shard 3)** using LLM-guided iterative refinement (OpenAI/Groq) backed by hybrid semantic embedding validation.

### 🛠️ Tech Stack
- **Language & Runtime**: Python 3.10+, Jupyter Notebooks.
- **LLM APIs & Automation**: OpenAI API (`gpt-4o-mini`), Groq API, batch processing pipelines with exponential backoff retries and JSONL checkpoint persistence.
- **NLP & Quality Control**: Sentence-Transformers, multilingual cross-lingual sentence embeddings, cosine similarity scoring, Label Studio schema (`Labeling_Interface.xml`).
- **Analytics & Academic Publishing**: Pandas, NumPy, Matplotlib, Seaborn, LaTeX & Beamer (`Rapport_QC_Cleaning_MT.pdf`, `Presentation_Projet_Traduction.pdf`).

### 👩‍💻 My Role & Key Contributions
- **Batch Correction Architecture (`correct_silver.py`)**:
  - Engineered a fault-tolerant batch translation engine processing 20-sample windows with dynamic prompt engineering tailored to Moroccan Darija idioms.
  - Built stateful checkpoints ensuring resilient execution during large-scale API batch calls.
- **Hybrid Semantic Consistency Engine (`hybrid_semantic_consistency.py`)**:
  - Implemented vector similarity checks flagging semantically drifted or hallucinated pairs.
  - Built automated pruning heuristics rejecting low-confidence translations below calibrated cosine boundaries.
- **Comparative Data Exploration (EDA)**:
  - Conducted statistical diagnostics assessing lexical diversity, length ratios, and error distributions comparing Gold Shard 3 with LLM-corrected Silver Shard 3.
- **Scientific Deliverables**:
  - Authored the technical methodology report (`Rapport_QC_Cleaning_MT.pdf`) and executive conference presentation slides (`Presentation_Projet_Traduction.pdf`).

### 📊 Key Results & Impact
- **Large-Scale Corpus Cleansing**: Successfully enhanced and standardized 7,000+ conversational pairs with >95% semantic preservation.
- **High-Precision Quality Gate**: Eliminated over 80% of literal translation artifacts via multilingual embedding filters.
- **Turnkey Academic Package**: Comprehensive documentation, LaTeX source code, and compiled presentation assets.

---

### 🚀 Quick Start / Démarrage Rapide

#### 1. Setup Environment
```bash
python -m venv env
# Windows:
env\Scripts\Activate.ps1
# Linux/macOS:
# source env/bin/activate
pip install -r requirements.txt
```

#### 2. Configuration
```bash
cp .env.example .env
# Add your OPENAI_API_KEY or GROQ_API_KEY in .env
```

#### 3. Run Semantic Correction & QC
```bash
# Run batch correction
python correct_silver.py

# Run hybrid semantic validation
python hybrid_semantic_consistency.py
```
