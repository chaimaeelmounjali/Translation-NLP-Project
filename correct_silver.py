#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de correction Silver Standard pour corpus Darija marocain
Traite les 5 shards CSV avec l'API Anthropic (claude-haiku-4-5-20251001)
Génère rapports détaillés et checkpoints de sauvegarde
"""

import os
import sys
import json
import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import random
import traceback

# Dépendances externes
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Charger les variables d'environnement
load_dotenv()

# Créer les répertoires de sortie
DIRS = {
    "corrected": Path("corrected_silver"),
    "reports": Path("reports"),
    "checkpoints": Path("checkpoints"),
}

for dir_path in DIRS.values():
    dir_path.mkdir(exist_ok=True)

# Configuration logging
log_file = Path("correction_errors.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paramètres API
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("Variable d'environnement OPENAI_API_KEY non configurée")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

# Chemins des fichiers
SHARDS_INPUT_DIR = Path("shards/silver_9k_shards")
SHARD_FILES = [SHARDS_INPUT_DIR / f"silver_shard_{i}.csv" for i in range(1, 6)]

# Paramètres
BATCH_SIZE = 10  # Batches de 10 pour meilleure stabilité JSON
CHECKPOINT_INTERVAL = 50  # Checkpoint tous les 50 lignes (5 batches) pour rapide feedback
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2
TEMPERATURE = 0.1
MAX_TOKENS = 4096
MODEL = "gpt-4o-mini"  # Modèle rapide et économe pour OpenAI

# ============================================================================
# SYSTÈME DE PROMPTS
# ============================================================================

SYSTEM_PROMPT = """Tu es un expert traducteur en Darija marocaine, Anglais et Arabe Standard (MSA).
Corrige uniquement les champs de traduction pour qu'ils soient sémantiquement
cohérents avec le champ english. 
IMPORTANT: Retourne UNIQUEMENT un tableau JSON valide en UTF-8, 
sans markdown, sans guillemets non-échappés, sans texte avant ou après le JSON.
Assure-toi que tous les caractères spéciaux (arabes, accents, etc.) sont correctement échappés."""

USER_PROMPT_TEMPLATE = """Règles :
- darija_arabic : script arabe uniquement (aucun caractère latin ni chiffre)
- darija_arabizi : translittération latine phonétique (3=ع, 7=ح, 9=ق, ch=ش)
- modern_standard_arabic : MSA formel fidèle à l'anglais
- status : toujours "VALIDATED" après correction
- Ne modifie JAMAIS les champs : data_id, id, classe, english, english_word_count

IMPORTANT: Retourne UNIQUEMENT du JSON valide, rien d'autre. Pas de texte avant ou après.
Les caractères arabes doivent être correctement échappés en UTF-8.

Pour chaque ligne de statut "PARTIALLY VALIDATED" :
  corriger UNIQUEMENT modern_standard_arabic.
Pour chaque ligne de statut "GENERATED" :
  corriger darija_arabic, darija_arabizi ET modern_standard_arabic.

Format de retour (tableau JSON STRICT) :
[{{"data_id": "...", "darija_arabic": "...", "darija_arabizi": "...", "modern_standard_arabic": "...", "status": "VALIDATED"}}]

Données à corriger :
{batch_json}"""

# ============================================================================
# UTILITAIRES
# ============================================================================

def retry_with_backoff(func, max_retries=MAX_RETRIES):
    """Décorateur de retry avec backoff exponentiel"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = RETRY_BASE_DELAY ** attempt
            logger.warning(
                f"Tentative {attempt + 1} échouée. "
                f"Nouvelle tentative dans {delay}s. Erreur: {str(e)}"
            )
            time.sleep(delay)

def detect_arabizi(text):
    """Détecte la présence de caractères Arabizi (chiffres 2,3,7,9) dans du texte"""
    if not isinstance(text, str):
        return False
    arabizi_chars = {'2', '3', '7', '9'}
    return any(c in text for c in arabizi_chars)

def is_empty_or_na(value):
    """Vérifie si une valeur est vide ou NA"""
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False

def call_openai_api(batch_data):
    """Appel à l'API OpenAI avec retry exponentiel"""
    batch_json = json.dumps(batch_data, ensure_ascii=False, indent=2)
    user_message = USER_PROMPT_TEMPLATE.format(batch_json=batch_json)
    
    def api_call():
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    
    response_text = retry_with_backoff(api_call)
    
    # Parser la réponse JSON avec meilleure gestion d'erreurs
    try:
        # Essayer de parser directement
        corrected_batch = json.loads(response_text)
        if not isinstance(corrected_batch, list):
            raise ValueError("Réponse API n'est pas un tableau JSON")
        return corrected_batch
    except json.JSONDecodeError as e:
        # Essayer de nettoyer la réponse (supprimer du texte avant/après)
        logger.warning(f"Première tentative de parsing échouée: {str(e)}")
        
        try:
            # Chercher le début et la fin du JSON
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx >= 0 and end_idx > start_idx:
                clean_json = response_text[start_idx:end_idx+1]
                corrected_batch = json.loads(clean_json)
                if isinstance(corrected_batch, list):
                    logger.info("Parsing JSON réussi après nettoyage")
                    return corrected_batch
        except:
            pass
        
        logger.error(f"Impossible de parser la réponse JSON: {response_text[:300]}")
        raise ValueError(f"Erreur de parsing JSON: {str(e)}")

# ============================================================================
# CLASSE DE TRAITEMENT
# ============================================================================

class SilverCorrectionProcessor:
    """Traite la correction d'un shard Silver"""
    
    def __init__(self, shard_index):
        self.shard_index = shard_index
        self.shard_file = SHARD_FILES[shard_index - 1]
        self.start_time = None
        self.stats = {
            "total_lines": 0,
            "validated_success": 0,
            "errors": 0,
            "by_input_status": defaultdict(lambda: {"total": 0, "corrected": 0, "errors": 0}),
            "by_class": defaultdict(lambda: {"total": 0, "corrected": 0, "errors": 0}),
            "corrections": {
                "darija_arabic": 0,
                "darija_arabizi": 0,
                "modern_standard_arabic": 0,
                "partial_corrections": 0,
                "complete_corrections": 0,
            },
            "anomalies": {
                "arabizi_before": 0,
                "arabizi_after": 0,
                "empty_before": 0,
                "empty_after": 0,
            },
            "error_lines": []
        }
        self.corrections_log = []  # Pour les exemples
    
    def load_data(self):
        """Charge le fichier CSV"""
        logger.info(f"Chargement du shard {self.shard_index} depuis {self.shard_file}")
        try:
            df = pd.read_csv(self.shard_file, encoding='utf-8')
            logger.info(f"Shard {self.shard_index}: {len(df)} lignes chargées")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {self.shard_file}: {str(e)}")
            raise
    
    def process(self):
        """Traite le shard complet"""
        self.start_time = datetime.now()
        logger.info(f"=== Début du traitement du shard {self.shard_index} ===")
        
        try:
            df = self.load_data()
            self.stats["total_lines"] = len(df)
            
            corrected_rows = []
            
            # Traiter par batches
            for batch_start in tqdm(
                range(0, len(df), BATCH_SIZE),
                desc=f"Shard {self.shard_index}",
                unit="batch"
            ):
                batch_end = min(batch_start + BATCH_SIZE, len(df))
                batch_rows = df.iloc[batch_start:batch_end]
                
                # Convertir le batch en dictionnaires
                batch_data = [
                    {
                        "data_id": row.get("data_id"),
                        "id": row.get("id"),
                        "classe": row.get("classe"),
                        "darija_arabic": row.get("darija_arabic"),
                        "darija_arabizi": row.get("darija_arabizi"),
                        "english": row.get("english"),
                        "modern_standard_arabic": row.get("modern_standard_arabic"),
                        "english_word_count": row.get("english_word_count"),
                        "status": row.get("status")
                    }
                    for _, row in batch_rows.iterrows()
                ]
                
                # Corriger le batch
                try:
                    corrected_batch = call_openai_api(batch_data)
                    self._process_corrected_batch(batch_data, corrected_batch, corrected_rows)
                except Exception as e:
                    logger.error(f"Erreur lors de la correction du batch {batch_start}-{batch_end}: {str(e)}")
                    for row in batch_data:
                        self.stats["errors"] += 1
                        self.stats["error_lines"].append({
                            "data_id": row.get("data_id"),
                            "reason": str(e)[:100]
                        })
                        corrected_rows.append(row)  # Garder l'original en cas d'erreur
                
                # Checkpoint tous les 100 lignes
                if (batch_end % CHECKPOINT_INTERVAL) == 0:
                    self._save_checkpoint(corrected_rows)
                    # Sauvegarder aussi le fichier de résultats courant
                    self._save_corrected_data(corrected_rows, is_checkpoint=True)
            
            # Sauvegarder le résultat final
            self._save_corrected_data(corrected_rows)
            
            logger.info(f"=== Fin du traitement du shard {self.shard_index} ===")
            return corrected_rows
        
        except Exception as e:
            logger.error(f"Erreur critique lors du traitement du shard {self.shard_index}: {str(e)}")
            raise
    
    def _process_corrected_batch(self, original_batch, corrected_batch, corrected_rows):
        """Traite les lignes corrigées et met à jour les stats"""
        for original_row, corrected_row in zip(original_batch, corrected_batch):
            try:
                data_id = original_row.get("data_id")
                input_status = original_row.get("status")
                classe = original_row.get("classe")
                
                # Mettre à jour les stats
                self.stats["by_input_status"][input_status]["total"] += 1
                self.stats["by_class"][classe]["total"] += 1
                
                # Compter les corrections
                fields_corrected = []
                
                if corrected_row.get("darija_arabic") != original_row.get("darija_arabic"):
                    self.stats["corrections"]["darija_arabic"] += 1
                    fields_corrected.append("darija_arabic")
                
                if corrected_row.get("darija_arabizi") != original_row.get("darija_arabizi"):
                    self.stats["corrections"]["darija_arabizi"] += 1
                    fields_corrected.append("darija_arabizi")
                
                if corrected_row.get("modern_standard_arabic") != original_row.get("modern_standard_arabic"):
                    self.stats["corrections"]["modern_standard_arabic"] += 1
                    fields_corrected.append("modern_standard_arabic")
                
                # Compter le type de correction
                if input_status == "GENERATED" and len(fields_corrected) > 0:
                    self.stats["corrections"]["complete_corrections"] += 1
                elif input_status == "PARTIALLY VALIDATED" and len(fields_corrected) > 0:
                    self.stats["corrections"]["partial_corrections"] += 1
                
                # Déterminer le statut
                corrected_row["status"] = "VALIDATED"
                
                # Conserver les champs immuables
                for immutable_field in ["data_id", "id", "classe", "english", "english_word_count"]:
                    corrected_row[immutable_field] = original_row.get(immutable_field)
                
                # Détecter les anomalies AVANT correction
                if detect_arabizi(original_row.get("darija_arabic")):
                    self.stats["anomalies"]["arabizi_before"] += 1
                
                for field in ["darija_arabic", "darija_arabizi", "modern_standard_arabic"]:
                    if is_empty_or_na(original_row.get(field)):
                        self.stats["anomalies"]["empty_before"] += 1
                
                # Détecter les anomalies APRÈS correction
                if detect_arabizi(corrected_row.get("darija_arabic")):
                    self.stats["anomalies"]["arabizi_after"] += 1
                
                for field in ["darija_arabic", "darija_arabizi", "modern_standard_arabic"]:
                    if is_empty_or_na(corrected_row.get(field)):
                        self.stats["anomalies"]["empty_after"] += 1
                
                # Mettre à jour les stats de classe et status
                self.stats["by_input_status"][input_status]["corrected"] += 1
                self.stats["by_class"][classe]["corrected"] += 1
                self.stats["validated_success"] += 1
                
                # Logger pour la génération d'exemples
                if len(self.corrections_log) < 10:  # Garder 10 exemples
                    for field in fields_corrected:
                        self.corrections_log.append({
                            "data_id": data_id,
                            "input_status": input_status,
                            "field": field,
                            "before": original_row.get(field),
                            "after": corrected_row.get(field)
                        })
                
                corrected_rows.append(corrected_row)
            
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la ligne {data_id}: {str(e)}")
                self.stats["errors"] += 1
                self.stats["error_lines"].append({
                    "data_id": original_row.get("data_id"),
                    "reason": str(e)[:100]
                })
                corrected_rows.append(original_row)  # Garder l'original
    
    def _save_checkpoint(self, corrected_rows):
        """Sauvegarde un checkpoint"""
        checkpoint_file = DIRS["checkpoints"] / f"shard_{self.shard_index}_checkpoint.csv"
        try:
            df = pd.DataFrame(corrected_rows)
            df.to_csv(checkpoint_file, index=False, encoding='utf-8')
            logger.info(f"Checkpoint sauvegardé: {checkpoint_file} ({len(corrected_rows)} lignes)")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du checkpoint: {str(e)}")
    
    def _save_corrected_data(self, corrected_rows, is_checkpoint=False):
        """Sauvegarde les données corrigées"""
        suffix = "_checkpoint_progress" if is_checkpoint else ""
        output_file = DIRS["corrected"] / f"corrected_silver_shard_{self.shard_index}{suffix}.csv"
        try:
            df = pd.DataFrame(corrected_rows)
            df.to_csv(output_file, index=False, encoding='utf-8')
            status = "intermédiaire" if is_checkpoint else "finale"
            logger.info(f"Données corrigées {status} sauvegardées: {output_file} ({len(corrected_rows)} lignes)")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données corrigées: {str(e)}")
    
    def generate_report(self):
        """Génère le rapport Markdown pour ce shard"""
        duration = (datetime.now() - self.start_time).total_seconds()
        duration_min = duration / 60
        
        report_file = DIRS["reports"] / f"correction_report_shard_{self.shard_index}.md"
        
        with open(report_file, "w", encoding='utf-8') as f:
            # EN-TÊTE
            f.write(f"# Rapport de Correction - Shard {self.shard_index}\n\n")
            f.write(f"**Fichier** : `silver_shard_{self.shard_index}.csv`  \n")
            f.write(f"**Date/Heure** : {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Durée totale** : {duration_min:.2f} minutes  \n\n")
            
            # RÉSUMÉ EXÉCUTIF
            f.write("## 📊 Résumé Exécutif\n\n")
            f.write(f"- **Lignes traitées** : {self.stats['total_lines']}\n")
            f.write(f"- **Lignes validées avec succès** : {self.stats['validated_success']}\n")
            f.write(f"- **Lignes en erreur** : {self.stats['errors']}\n")
            
            if self.stats['total_lines'] > 0:
                taux_correction = (self.stats['validated_success'] / self.stats['total_lines']) * 100
                f.write(f"- **Taux de correction** : {taux_correction:.2f}%\n\n")
            
            # STATISTIQUES PAR STATUT D'ENTRÉE
            f.write("## 📈 Statistiques par Statut d'Entrée\n\n")
            f.write("| Statut | Nb Lignes | % Total | Nb Corrigées | Nb Erreurs |\n")
            f.write("|--------|-----------|---------|--------------|------------|\n")
            
            for status, stats in self.stats["by_input_status"].items():
                pct = (stats['total'] / self.stats['total_lines'] * 100) if self.stats['total_lines'] > 0 else 0
                f.write(
                    f"| {status} | {stats['total']} | {pct:.1f}% | "
                    f"{stats['corrected']} | {stats['errors']} |\n"
                )
            
            f.write("\n")
            
            # STATISTIQUES PAR CLASSE
            f.write("## 🏷 Statistiques par Classe\n\n")
            f.write("| Classe | Nb Lignes | Nb Corrigées | Nb Erreurs |\n")
            f.write("|--------|-----------|--------------|------------|\n")
            
            for classe in sorted(self.stats["by_class"].keys()):
                stats = self.stats["by_class"][classe]
                f.write(
                    f"| {classe} | {stats['total']} | {stats['corrected']} | "
                    f"{stats['errors']} |\n"
                )
            
            f.write("\n")
            
            # TYPES DE CORRECTIONS
            f.write("## 🔧 Types de Corrections Effectuées\n\n")
            f.write(f"- **darija_arabic corrigés** : {self.stats['corrections']['darija_arabic']}\n")
            f.write(f"- **darija_arabizi corrigés** : {self.stats['corrections']['darija_arabizi']}\n")
            f.write(f"- **modern_standard_arabic corrigés** : {self.stats['corrections']['modern_standard_arabic']}\n")
            f.write(f"- **Corrections partielles** (PARTIALLY VALIDATED → MSA) : {self.stats['corrections']['partial_corrections']}\n")
            f.write(f"- **Corrections complètes** (GENERATED → 3 champs) : {self.stats['corrections']['complete_corrections']}\n\n")
            
            # ANOMALIES
            f.write("## ⚠️ Détection d'Anomalies\n\n")
            f.write("### Avant Correction\n\n")
            f.write(f"- Lignes avec Arabizi dans darija_arabic : {self.stats['anomalies']['arabizi_before']}\n")
            f.write(f"- Champs vides : {self.stats['anomalies']['empty_before']}\n\n")
            
            f.write("### Après Correction\n\n")
            f.write(f"- Lignes avec Arabizi dans darija_arabic : {self.stats['anomalies']['arabizi_after']}\n")
            f.write(f"- Champs vides : {self.stats['anomalies']['empty_after']}\n\n")
            
            # EXEMPLES DE CORRECTIONS
            f.write("## 📝 Exemples de Corrections\n\n")
            if self.corrections_log:
                f.write("| data_id | Statut Initial | Champ Modifié | Avant | Après |\n")
                f.write("|---------|----------------|---------------|-------|-------|\n")
                
                for example in self.corrections_log[:10]:
                    before_short = str(example['before'])[:30]
                    after_short = str(example['after'])[:30]
                    f.write(
                        f"| {example['data_id']} | {example['input_status']} | "
                        f"{example['field']} | {before_short} | {after_short} |\n"
                    )
                f.write("\n")
            else:
                f.write("Aucun exemple disponible.\n\n")
            
            # LIGNES EN ERREUR
            f.write("## ❌ Lignes en Erreur\n\n")
            if self.stats['error_lines']:
                f.write("| data_id | Raison |\n")
                f.write("|---------|--------|\n")
                for error in self.stats['error_lines'][:20]:
                    f.write(f"| {error['data_id']} | {error['reason']} |\n")
            else:
                f.write("Aucune erreur détectée.\n")
        
        logger.info(f"Rapport généré: {report_file}")
        return report_file

# ============================================================================
# RAPPORT GLOBAL
# ============================================================================

class GlobalReporter:
    """Génère le rapport global après traitement de tous les shards"""
    
    def __init__(self):
        self.shards_stats = {}
        self.global_start_time = datetime.now()
    
    def add_shard_stats(self, shard_index, stats):
        """Ajoute les stats d'un shard"""
        self.shards_stats[shard_index] = stats
    
    def generate(self):
        """Génère le rapport global"""
        report_file = DIRS["reports"] / "global_correction_report.md"
        
        # Calculer les statistiques globales
        total_lines = sum(s['total_lines'] for s in self.shards_stats.values())
        total_validated = sum(s['validated_success'] for s in self.shards_stats.values())
        total_errors = sum(s['errors'] for s in self.shards_stats.values())
        
        duration = (datetime.now() - self.global_start_time).total_seconds()
        duration_hours = duration / 3600
        
        with open(report_file, "w", encoding='utf-8') as f:
            # EN-TÊTE
            f.write("# Rapport Global de Correction - Silver Standard\n\n")
            f.write(f"**Date/Heure** : {self.global_start_time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Durée totale** : {duration_hours:.2f} heures  \n")
            f.write(f"**Nombre de shards traités** : {len(self.shards_stats)}\n\n")
            
            # RÉSUMÉ GLOBAL
            f.write("## 🎯 Résumé Global\n\n")
            f.write(f"- **Total lignes traitées** : {total_lines}\n")
            f.write(f"- **Total VALIDATED** : {total_validated}\n")
            f.write(f"- **Total erreurs** : {total_errors}\n")
            
            if total_lines > 0:
                taux_global = (total_validated / total_lines) * 100
                f.write(f"- **Taux de correction global** : {taux_global:.2f}%\n\n")
            
            # TABLEAU DE COMPARAISON PAR SHARD
            f.write("## 📊 Comparaison entre Shards\n\n")
            f.write("| Shard | Lignes | VALIDATED | Erreurs | Taux (%) |\n")
            f.write("|-------|--------|-----------|---------|----------|\n")
            
            for shard_idx in sorted(self.shards_stats.keys()):
                stats = self.shards_stats[shard_idx]
                taux = (stats['validated_success'] / stats['total_lines'] * 100) if stats['total_lines'] > 0 else 0
                f.write(
                    f"| {shard_idx} | {stats['total_lines']} | "
                    f"{stats['validated_success']} | {stats['errors']} | {taux:.2f}% |\n"
                )
            
            f.write("\n")
            
            # DISTRIBUTION GLOBALE PAR STATUT
            f.write("## 🏛 Distribution Globale par Statut d'Entrée\n\n")
            f.write("| Statut | Nb Lignes | % du total | Nb Corrigées |\n")
            f.write("|--------|-----------|-----------|-------------|\n")
            
            global_by_status = defaultdict(lambda: {"total": 0, "corrected": 0})
            for stats in self.shards_stats.values():
                for status, stat_dict in stats["by_input_status"].items():
                    global_by_status[status]["total"] += stat_dict["total"]
                    global_by_status[status]["corrected"] += stat_dict["corrected"]
            
            for status in sorted(global_by_status.keys()):
                pct = (global_by_status[status]['total'] / total_lines * 100) if total_lines > 0 else 0
                f.write(
                    f"| {status} | {global_by_status[status]['total']} | {pct:.1f}% | "
                    f"{global_by_status[status]['corrected']} |\n"
                )
            
            f.write("\n")
            
            # DISTRIBUTION GLOBALE PAR CLASSE
            f.write("## 📋 Distribution Globale par Classe\n\n")
            f.write("| Classe | Nb Lignes | Nb Corrigées | Nb Erreurs |\n")
            f.write("|--------|-----------|--------------|------------|\n")
            
            global_by_class = defaultdict(lambda: {"total": 0, "corrected": 0, "errors": 0})
            for stats in self.shards_stats.values():
                for classe, stat_dict in stats["by_class"].items():
                    global_by_class[classe]["total"] += stat_dict["total"]
                    global_by_class[classe]["corrected"] += stat_dict["corrected"]
                    global_by_class[classe]["errors"] += stat_dict["errors"]
            
            for classe in sorted(global_by_class.keys()):
                f.write(
                    f"| {classe} | {global_by_class[classe]['total']} | "
                    f"{global_by_class[classe]['corrected']} | {global_by_class[classe]['errors']} |\n"
                )
            
            f.write("\n")
            
            # STATISTIQUES GLOBALES DE CORRECTIONS
            f.write("## 🔧 Statistiques Globales de Corrections\n\n")
            
            total_darija_arabic = sum(s['corrections']['darija_arabic'] for s in self.shards_stats.values())
            total_darija_arabizi = sum(s['corrections']['darija_arabizi'] for s in self.shards_stats.values())
            total_msa = sum(s['corrections']['modern_standard_arabic'] for s in self.shards_stats.values())
            total_partial = sum(s['corrections']['partial_corrections'] for s in self.shards_stats.values())
            total_complete = sum(s['corrections']['complete_corrections'] for s in self.shards_stats.values())
            
            f.write(f"- **darija_arabic corrigés** : {total_darija_arabic}\n")
            f.write(f"- **darija_arabizi corrigés** : {total_darija_arabizi}\n")
            f.write(f"- **modern_standard_arabic corrigés** : {total_msa}\n")
            f.write(f"- **Corrections partielles** : {total_partial}\n")
            f.write(f"- **Corrections complètes** : {total_complete}\n")
        
        logger.info(f"Rapport global généré: {report_file}")

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Point d'entrée principal"""
    logger.info("[START] Démarrage du script de correction Silver Standard")
    logger.info(f"Traitement de shards depuis: {SHARDS_INPUT_DIR}")
    
    # Vérifier que les fichiers existent
    for shard_file in SHARD_FILES:
        if not shard_file.exists():
            logger.error(f"Fichier non trouvé: {shard_file}")
            sys.exit(1)
    
    global_reporter = GlobalReporter()
    
    try:
        for shard_index in range(1, 6):
            processor = SilverCorrectionProcessor(shard_index)
            try:
                processor.process()
                processor.generate_report()
                global_reporter.add_shard_stats(shard_index, processor.stats)
            except Exception as e:
                logger.error(f"Erreur lors du traitement du shard {shard_index}: {str(e)}")
                traceback.print_exc()
        
        # Générer le rapport global
        global_reporter.generate()
        
        logger.info(f"✅ Tous les shards ont été traités avec succès!")
        logger.info(f"Résultats sauvegardés dans:")
        logger.info(f"  - Données: {DIRS['corrected']}")
        logger.info(f"  - Rapports: {DIRS['reports']}")
        logger.info(f"  - Checkpoints: {DIRS['checkpoints']}")
        logger.info(f"  - Erreurs: {log_file}")
    
    except Exception as e:
        logger.error(f"Erreur critique: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
