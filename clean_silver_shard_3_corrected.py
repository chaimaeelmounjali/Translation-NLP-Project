#!/usr/bin/env python3
"""
Production-ready NLP cleaning pipeline for corrected silver_shard_3 data.

This script demonstrates the pipeline with a dummy DataFrame, then applies
the same processing to a CSV file containing at least these columns:
  - darija_arabic
  - darija_arabizi
  - english
  - modern_standard_arabic
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


TARGET_COLUMNS = [
    "darija_arabic",
    "darija_arabizi",
    "english",
    "modern_standard_arabic",
]
QC_COLUMNS = ("qc_changed_fields", "qc_notes")

GLOBAL_ARTIFACT_PATTERN = re.compile(r"\s*(?:<unk>|@-@)\s*")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
ISOLATED_DIGITS_PATTERN = re.compile(r"(?<![a-zA-Z])[0-9]+(?![a-zA-Z])")
LATIN_ALNUM_SPACE_PATTERN = re.compile(r"[^a-z0-9\s]")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
MARKDOWN_INLINE_CODE_PATTERN = re.compile(r"`{1,3}[^`]+`{1,3}")
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+"
)
TARGET_COLUMNS_COMMA = ",".join(TARGET_COLUMNS).lower()
TARGET_COLUMNS_SEMICOLON = ";".join(TARGET_COLUMNS).lower()
METADATA_ACTION_KEYWORDS = (
    "corrected",
    "removed",
    "adjusted",
    "fixed",
    "improved",
    "faithful",
    "natural",
    "mistranslation",
    "spelling",
)
METADATA_LANGUAGE_KEYWORDS = (
    "darija",
    "arabizi",
    "english",
    "msa",
    "modern standard arabic",
    "mt artifact",
    "mt artifacts",
)
DARIJA_LEAKAGE_MARKERS = {
    "ديال",
    "بزاف",
    "واش",
    "غادي",
    "حيت",
    "كاين",
    "مازال",
    "دابا",
    "هاد",
    "داك",
    "علاش",
    "فين",
    "باش",
    "حنا",
}
PLACEHOLDER_TERMS_BY_COLUMN = {
    "darija_arabic": {
        "كلمه غىر مفهومه",
        "كلمة غير مفهومة",
        "اسم قديم",
        "اسم جديد",
    },
    "darija_arabizi": {
        "kalima ghair mafhuma",
        "ism qadim",
        "ism jadid",
    },
    "english": {
        "kalima ghair mafhuma",
    },
    "modern_standard_arabic": {
        "كلمه غىر مفهومه",
        "كلمة غير مفهومة",
        "اسم قديم",
        "اسم جديد",
    },
}

# Arabic diacritics (Tashkeel) and related marks.
ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL_PATTERN = re.compile(r"\u0640")
ALEF_VARIANTS_PATTERN = re.compile(r"[\u0622\u0623\u0625\u0671]")
NON_ARABIC_PATTERN = re.compile(r"[^\u0621-\u063A\u0641-\u064A\u0649\s]")

DEFAULT_DARIJA_STOPWORDS = {
    "ف",
    "في",
    "من",
    "على",
    "و",
    "يا",
    "هاد",
    "داك",
    "ديال",
    "ما",
    "كي",
    "باش",
    "مع",
    "الى",
    "إلى",
}

DEFAULT_ARABIZI_STOPWORDS = {
    "f",
    "fi",
    "men",
    "mn",
    "w",
    "dyal",
    "dial",
    "m3a",
    "ma",
    "ila",
    "l",
    "b",
    "3la",
}

FALLBACK_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "this",
    "these",
    "those",
    "or",
    "if",
    "but",
    "than",
    "then",
    "there",
    "their",
    "they",
    "them",
    "we",
    "our",
    "you",
    "your",
    "i",
    "me",
    "my",
    "mine",
    "she",
    "her",
    "his",
    "him",
    "not",
    "no",
    "do",
    "does",
    "did",
    "have",
    "had",
    "been",
}

FALLBACK_ARABIC_STOPWORDS = {
    "في",
    "من",
    "على",
    "الى",
    "إلى",
    "عن",
    "ما",
    "لا",
    "لم",
    "لن",
    "هو",
    "هي",
    "هم",
    "هن",
    "هذا",
    "هذه",
    "ذلك",
    "تلك",
    "هناك",
    "ثم",
    "او",
    "أو",
    "اذا",
    "إذا",
    "كان",
    "كانت",
    "يكون",
    "يمكن",
    "قد",
    "لقد",
}

ENGLISH_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
}

DARIJA_VARIANT_MAP = {
    "دىل": "دىال",
    "دىلنا": "دىالنا",
    "ديل": "ديال",
    "ديلنا": "ديالنا",
    "بزااف": "بزاف",
    "بززاف": "بزاف",
    "علاش": "علاش",
}

ARABIC_PROCLITICS = (
    "وال",
    "بال",
    "كال",
    "فال",
    "لل",
    "ال",
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
    "س",
)
ARABIC_SINGLE_PREFIX_HINTS = set("المنيتس")
ARABIC_ENCLITICS = (
    "كما",
    "كم",
    "كن",
    "هما",
    "هم",
    "هن",
    "ها",
    "نا",
    "ني",
    "ه",
    "ك",
    "ي",
)

ARABIZI_DIGIT_TO_ARABIC = {
    "2": "ء",
    "3": "ع",
    "5": "خ",
    "7": "ح",
    "8": "غ",
    "9": "ق",
}

ARABIZI_DIGRAPH_TO_ARABIC = {
    "kh": "خ",
    "gh": "غ",
    "ch": "ش",
    "sh": "ش",
    "th": "ث",
    "dh": "ذ",
}

ARABIZI_CHAR_TO_ARABIC = {
    "a": "ا",
    "b": "ب",
    "c": "ك",
    "d": "د",
    "e": "ي",
    "f": "ف",
    "g": "ج",
    "h": "ه",
    "i": "ي",
    "j": "ج",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "o": "و",
    "p": "ب",
    "q": "ق",
    "r": "ر",
    "s": "س",
    "t": "ت",
    "u": "و",
    "v": "ف",
    "w": "و",
    "x": "خ",
    "y": "ي",
    "z": "ز",
}


@dataclass(frozen=True)
class CleaningOptions:
    english_normalization: str = "none"
    expand_english_contractions: bool = True
    split_arabic_clitics: bool = False
    split_darija_negation: bool = False
    use_light_arabic_stemming: bool = False
    transliterate_arabizi_to_arabic: bool = False
    arabizi_drop_vowels: bool = False
    arabizi_max_char_repeat: int = 2
    emoji_replacement: str = " "


def load_stopwords_or_fallback(language: str, fallback_values: Iterable[str]) -> Set[str]:
    """Load NLTK stopwords; fallback to local defaults if unavailable."""
    try:
        return set(stopwords.words(language))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words(language))
        except Exception:
            return set(fallback_values)


def load_wordnet_lemmatizer_if_available() -> WordNetLemmatizer | None:
    """Load WordNet resources; return None if unavailable."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            return None
    try:
        return WordNetLemmatizer()
    except Exception:
        return None


def as_text(value: object) -> str:
    """Safely convert missing/non-string values to a normalized string."""
    if pd.isna(value):
        return ""
    return str(value)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def strip_html_tags(text: str) -> str:
    return HTML_TAG_PATTERN.sub(" ", text)


def strip_markdown(text: str) -> str:
    cleaned = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    cleaned = MARKDOWN_INLINE_CODE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"(^|\s)[>#]+", " ", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    cleaned = cleaned.replace("~", " ")
    return cleaned


def remove_mentions(text: str) -> str:
    return MENTION_PATTERN.sub(" ", text)


def replace_emojis(text: str, replacement: str = " ") -> str:
    return EMOJI_PATTERN.sub(replacement, text)


def remove_global_artifacts(text: object) -> str:
    """Remove exact machine artifacts globally from all target columns."""
    raw = as_text(text)
    cleaned = GLOBAL_ARTIFACT_PATTERN.sub(" ", raw)
    return normalize_whitespace(cleaned)


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def strip_unicode_punctuation(text: str) -> str:
    return "".join(
        " " if unicodedata.category(char)[0] in {"P", "S"} else char
        for char in text
    )


def strip_non_latin_alnum(text: str) -> str:
    return LATIN_ALNUM_SPACE_PATTERN.sub(" ", text)


def apply_universal_text_cleanup(text: object, emoji_replacement: str = " ") -> str:
    cleaned = as_text(text)
    cleaned = html.unescape(cleaned)
    cleaned = remove_global_artifacts(cleaned)
    cleaned = strip_html_tags(cleaned)
    cleaned = strip_markdown(cleaned)
    cleaned = remove_urls(cleaned)
    cleaned = remove_mentions(cleaned)
    cleaned = replace_emojis(cleaned, replacement=emoji_replacement)
    return normalize_whitespace(cleaned)


def normalize_arabic_text(
    text: object,
    fold_ta_marbuta: bool,
    fold_yeh: bool,
) -> str:
    """
    Arabic normalization for MSA and Darija Arabic script.
    - remove diacritics and tatweel
    - Alef variants -> bare Alef
    - optional Ta marbuta -> Ha folding
    - optional Yaa -> Alef Maksura folding
    - remove non-Arabic chars and punctuation
    """
    normalized = as_text(text)
    normalized = ARABIC_DIACRITICS_PATTERN.sub("", normalized)
    normalized = TATWEEL_PATTERN.sub("", normalized)
    normalized = ALEF_VARIANTS_PATTERN.sub("ا", normalized)
    if fold_ta_marbuta:
        normalized = normalized.replace("ة", "ه")
    if fold_yeh:
        normalized = normalized.replace("ي", "ى")
    normalized = NON_ARABIC_PATTERN.sub(" ", normalized)
    return normalize_whitespace(normalized)


def normalize_arabic_stopwords(
    words: Iterable[str],
    fold_ta_marbuta: bool,
    fold_yeh: bool,
) -> Set[str]:
    normalized_words = set()
    for word in words:
        token = normalize_arabic_text(
            word,
            fold_ta_marbuta=fold_ta_marbuta,
            fold_yeh=fold_yeh,
        )
        if token:
            normalized_words.add(token)
    return normalized_words


def normalize_darija_variants(text: str) -> str:
    tokens = text.split()
    normalized_tokens = [DARIJA_VARIANT_MAP.get(tok, tok) for tok in tokens]
    return " ".join(normalized_tokens)


def split_arabic_clitics(text: str, split_negation: bool = False) -> str:
    segmented: List[str] = []
    for token in text.split():
        current = token

        if split_negation and current.startswith("ما") and current.endswith("ش") and len(current) >= 5:
            core = current[2:-1].strip()
            segmented.append("ما")
            if core:
                segmented.append(core)
            segmented.append("ش")
            continue

        prefixes: List[str] = []
        for candidate in ARABIC_PROCLITICS:
            if not current.startswith(candidate):
                continue
            if len(current) - len(candidate) < 3:
                continue
            if len(candidate) == 1 and len(current) >= 2 and current[1] not in ARABIC_SINGLE_PREFIX_HINTS:
                continue

            prefixes.append(candidate)
            current = current[len(candidate) :]
            break

        suffix = ""
        for candidate in ARABIC_ENCLITICS:
            if len(current) - len(candidate) >= 2 and current.endswith(candidate):
                suffix = candidate
                current = current[: -len(candidate)]
                break

        segmented.extend(prefixes)
        if current:
            segmented.append(current)
        if suffix:
            segmented.append(suffix)

    return " ".join(segmented)


def light_arabic_stem_token(token: str) -> str:
    prefixes = ("ال", "وال", "بال", "كال", "فال", "لل", "و", "ف", "ب", "ك", "ل")
    suffixes = ("كما", "كم", "كن", "هما", "هم", "هن", "ها", "نا", "ني", "ات", "ون", "ين", "ة", "ه")

    stemmed = token
    for pref in prefixes:
        if stemmed.startswith(pref) and len(stemmed) - len(pref) >= 3:
            stemmed = stemmed[len(pref) :]
            break
    for suf in suffixes:
        if stemmed.endswith(suf) and len(stemmed) - len(suf) >= 3:
            stemmed = stemmed[: -len(suf)]
            break
    return stemmed


def light_arabic_stem_tokens(tokens: Iterable[str]) -> List[str]:
    return [light_arabic_stem_token(tok) for tok in tokens]


def reduce_repeated_characters(text: str, max_repeats: int = 2) -> str:
    if max_repeats < 1:
        max_repeats = 1
    pattern = re.compile(r"(.)\1{" + str(max_repeats) + r",}")
    return pattern.sub(lambda match: match.group(1) * max_repeats, text)


def drop_arabizi_vowels(text: str) -> str:
    return re.sub(r"[aeiou]", "", text)


def transliterate_arabizi_token_to_arabic(token: str) -> str:
    lowered = token.lower()
    for digraph, replacement in ARABIZI_DIGRAPH_TO_ARABIC.items():
        lowered = lowered.replace(digraph, replacement)

    output_chars: List[str] = []
    for char in lowered:
        if char in ARABIZI_DIGIT_TO_ARABIC:
            output_chars.append(ARABIZI_DIGIT_TO_ARABIC[char])
        elif char in ARABIZI_CHAR_TO_ARABIC:
            output_chars.append(ARABIZI_CHAR_TO_ARABIC[char])
        else:
            output_chars.append(char)

    return "".join(output_chars)


def transliterate_arabizi_to_arabic(text: str) -> str:
    tokens = [transliterate_arabizi_token_to_arabic(tok) for tok in text.split()]
    return " ".join(tokens)


def expand_english_contractions(text: str) -> str:
    expanded = text
    for contraction, replacement in sorted(ENGLISH_CONTRACTIONS.items(), key=lambda item: -len(item[0])):
        expanded = re.sub(rf"\\b{re.escape(contraction)}\\b", replacement, expanded)
    return expanded


def has_metadata_keywords(text: str) -> bool:
    has_action = any(keyword in text for keyword in METADATA_ACTION_KEYWORDS)
    has_language = any(keyword in text for keyword in METADATA_LANGUAGE_KEYWORDS)
    return has_action and has_language


def looks_like_metadata_value(value: object) -> bool:
    text = normalize_whitespace(as_text(value)).lower()
    if not text:
        return False
    if text in {TARGET_COLUMNS_COMMA, TARGET_COLUMNS_SEMICOLON}:
        return True
    if text.count(";") >= 3 and all(column in text for column in TARGET_COLUMNS):
        return True
    if sum(column in text for column in TARGET_COLUMNS) >= 3:
        return True
    if len(text.split()) >= 6 and has_metadata_keywords(text):
        return True
    return False


def contains_darija_marker(text: object) -> bool:
    tokens = set(normalize_whitespace(as_text(text)).split())
    return any(marker in tokens for marker in DARIJA_LEAKAGE_MARKERS)


def remove_placeholder_terms(text: str, placeholder_terms: Set[str]) -> Tuple[str, int]:
    cleaned = text
    removed_count = 0
    for term in placeholder_terms:
        pattern = re.compile(rf"(^|\s){re.escape(term)}(\s|$)", flags=re.IGNORECASE)
        cleaned, replacements = pattern.subn(" ", cleaned)
        removed_count += replacements
    return normalize_whitespace(cleaned), removed_count


def collapse_adjacent_duplicate_tokens(text: str) -> Tuple[str, int]:
    tokens = text.split()
    if not tokens:
        return "", 0

    collapsed = [tokens[0]]
    removed = 0
    for token in tokens[1:]:
        # Skip suspicious adjacent duplicates but keep very short tokens
        # because some abbreviations are legitimately repeated.
        if token == collapsed[-1] and len(token) >= 3 and not token.isdigit():
            removed += 1
            continue
        collapsed.append(token)
    return " ".join(collapsed), removed


def apply_post_clean_quality_fixes(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    output = df.copy()
    placeholder_removals = 0
    duplicate_token_removals = 0

    for col in columns:
        placeholders = PLACEHOLDER_TERMS_BY_COLUMN.get(col, set())
        fixed_values: List[str] = []

        for raw in output[col].fillna("").astype(str):
            normalized = normalize_whitespace(raw)
            normalized, removed_placeholders = remove_placeholder_terms(normalized, placeholders)
            normalized, removed_duplicates = collapse_adjacent_duplicate_tokens(normalized)
            fixed_values.append(normalized)
            placeholder_removals += removed_placeholders
            duplicate_token_removals += removed_duplicates

        output[col] = fixed_values

    english_word_count_changed_rows = 0
    if "english_word_count" in output.columns:
        recalculated = output["english"].fillna("").astype(str).str.split().apply(len).astype(int)
        previous = pd.to_numeric(output["english_word_count"], errors="coerce").fillna(-1).astype(int)
        english_word_count_changed_rows = int((previous != recalculated).sum())
        output["english_word_count"] = recalculated

    stats = {
        "placeholder_removals": placeholder_removals,
        "duplicate_token_removals": duplicate_token_removals,
        "english_word_count_recomputed_rows": english_word_count_changed_rows,
    }
    return output, stats


def clean_english(
    text: object,
    english_stopwords: Set[str],
    options: CleaningOptions,
    stemmer: PorterStemmer,
    lemmatizer: WordNetLemmatizer | None,
) -> str:
    cleaned = as_text(text).lower()
    if options.expand_english_contractions:
        cleaned = expand_english_contractions(cleaned)
    cleaned = strip_unicode_punctuation(cleaned)
    cleaned = strip_non_latin_alnum(cleaned)
    tokens = [tok for tok in normalize_whitespace(cleaned).split() if tok not in english_stopwords]

    if options.english_normalization == "stem":
        tokens = [stemmer.stem(tok) for tok in tokens]
    elif options.english_normalization == "lemma" and lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return " ".join(tokens)


def clean_modern_standard_arabic(
    text: object,
    msa_stopwords: Set[str],
    options: CleaningOptions,
) -> str:
    cleaned = normalize_arabic_text(
        text,
        fold_ta_marbuta=False,
        fold_yeh=False,
    )

    if options.split_arabic_clitics:
        cleaned = split_arabic_clitics(cleaned, split_negation=False)

    tokens = [tok for tok in cleaned.split() if tok not in msa_stopwords]

    if options.use_light_arabic_stemming:
        tokens = light_arabic_stem_tokens(tokens)

    return " ".join(tokens)


def clean_darija_arabic(
    text: object,
    darija_stopwords: Set[str],
    options: CleaningOptions,
) -> str:
    """Darija Arabic cleaning with caller-provided custom stopwords."""
    cleaned = normalize_arabic_text(
        text,
        fold_ta_marbuta=True,
        fold_yeh=True,
    )
    cleaned = normalize_darija_variants(cleaned)

    if options.split_arabic_clitics:
        cleaned = split_arabic_clitics(cleaned, split_negation=options.split_darija_negation)

    tokens = [tok for tok in cleaned.split() if tok not in darija_stopwords]

    if options.use_light_arabic_stemming:
        tokens = light_arabic_stem_tokens(tokens)

    return " ".join(tokens)


def clean_darija_arabizi(
    text: object,
    arabizi_stopwords: Set[str],
    darija_stopwords_arabic: Set[str],
    options: CleaningOptions,
) -> str:
    """
    Arabizi cleaning that preserves digits attached to letters.
    Isolated digits are removed via regex lookarounds.
    """
    cleaned = as_text(text).lower()
    cleaned = reduce_repeated_characters(cleaned, max_repeats=options.arabizi_max_char_repeat)

    if options.arabizi_drop_vowels:
        cleaned = drop_arabizi_vowels(cleaned)

    if options.transliterate_arabizi_to_arabic:
        cleaned = transliterate_arabizi_to_arabic(cleaned)
        cleaned = normalize_arabic_text(cleaned, fold_ta_marbuta=True, fold_yeh=True)
        tokens = [tok for tok in cleaned.split() if tok not in darija_stopwords_arabic]
        if options.use_light_arabic_stemming:
            tokens = light_arabic_stem_tokens(tokens)
        return " ".join(tokens)

    cleaned = strip_unicode_punctuation(cleaned)
    cleaned = ISOLATED_DIGITS_PATTERN.sub(" ", cleaned)
    cleaned = strip_non_latin_alnum(cleaned)
    tokens = [tok for tok in normalize_whitespace(cleaned).split() if tok not in arabizi_stopwords]
    return " ".join(tokens)


def discover_top_darija_words(
    darija_series: pd.Series,
    excluded_stopwords: Iterable[str],
    top_n: int = 100,
) -> List[Tuple[str, int]]:
    """Find top frequent Darija Arabic words after excluding custom stopwords."""
    excluded = set(excluded_stopwords)
    counter: Counter[str] = Counter()

    for text in darija_series.fillna("").astype(str):
        tokens = [tok for tok in text.split() if tok and tok not in excluded]
        counter.update(tokens)

    return counter.most_common(top_n)


def validate_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def apply_global_preprocessing(
    df: pd.DataFrame,
    columns: Sequence[str],
    options: CleaningOptions,
) -> pd.DataFrame:
    """Apply global artifact cleanup to all target columns using Series.apply."""
    output = df.copy()
    for col in columns:
        output[col] = output[col].apply(
            lambda value: apply_universal_text_cleanup(value, emoji_replacement=options.emoji_replacement)
        )
    return output


def drop_rows_with_empty_targets(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, int]:
    """Drop rows where any target column is empty/whitespace after cleaning."""
    normalized = df.copy()
    for col in columns:
        normalized[col] = normalized[col].fillna("").astype(str)

    keep_mask = normalized[list(columns)].apply(
        lambda series: series.str.strip().ne("")
    ).all(axis=1)

    dropped_count = int((~keep_mask).sum())
    return normalized.loc[keep_mask].copy(), dropped_count


def drop_rows_with_metadata_contamination(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, int]:
    contamination_mask = df[list(columns)].apply(
        lambda series: series.apply(looks_like_metadata_value)
    ).any(axis=1)
    dropped_count = int(contamination_mask.sum())
    return df.loc[~contamination_mask].copy(), dropped_count


def drop_rows_with_msa_darija_leakage(
    df: pd.DataFrame,
    msa_column: str = "modern_standard_arabic",
) -> Tuple[pd.DataFrame, int]:
    leakage_mask = df[msa_column].apply(contains_darija_marker)
    dropped_count = int(leakage_mask.sum())
    return df.loc[~leakage_mask].copy(), dropped_count


def drop_qc_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [col for col in QC_COLUMNS if col in df.columns]
    if not columns_to_drop:
        return df
    return df.drop(columns=columns_to_drop)


def drop_rows_by_status(
    df: pd.DataFrame,
    statuses_to_drop: Set[str],
    status_column: str = "status",
) -> Tuple[pd.DataFrame, int]:
    if status_column not in df.columns:
        return df.copy(), 0

    normalized_statuses = {status.upper() for status in statuses_to_drop}
    drop_mask = df[status_column].fillna("").astype(str).str.upper().isin(normalized_statuses)
    dropped_count = int(drop_mask.sum())
    return df.loc[~drop_mask].copy(), dropped_count


def clean_parallel_corpus(
    df: pd.DataFrame,
    darija_stopwords: Iterable[str],
    arabizi_stopwords: Iterable[str],
    options: CleaningOptions,
    keep_qc_columns: bool = False,
    drop_msa_darija_leakage: bool = True,
    drop_error_status_rows: bool = True,
) -> Tuple[pd.DataFrame, List[Tuple[str, int]], Dict[str, int]]:
    """Run the full cleaning pipeline over the four MT columns."""
    validate_required_columns(df, TARGET_COLUMNS)

    english_sw = load_stopwords_or_fallback("english", FALLBACK_ENGLISH_STOPWORDS)
    arabic_sw = load_stopwords_or_fallback("arabic", FALLBACK_ARABIC_STOPWORDS)
    msa_sw = normalize_arabic_stopwords(
        arabic_sw,
        fold_ta_marbuta=False,
        fold_yeh=False,
    )
    darija_sw = normalize_arabic_stopwords(
        darija_stopwords,
        fold_ta_marbuta=True,
        fold_yeh=True,
    )
    arabizi_sw = set(arabizi_stopwords)
    stemmer = PorterStemmer()
    lemmatizer = load_wordnet_lemmatizer_if_available() if options.english_normalization == "lemma" else None

    output = apply_global_preprocessing(df, TARGET_COLUMNS, options)

    output["english"] = output["english"].apply(
        lambda val: clean_english(
            val,
            english_sw,
            options,
            stemmer,
            lemmatizer,
        )
    )
    output["modern_standard_arabic"] = output["modern_standard_arabic"].apply(
        lambda val: clean_modern_standard_arabic(val, msa_sw, options)
    )
    output["darija_arabic"] = output["darija_arabic"].apply(
        lambda val: clean_darija_arabic(val, darija_sw, options)
    )
    output["darija_arabizi"] = output["darija_arabizi"].apply(
        lambda val: clean_darija_arabizi(
            val,
            arabizi_sw,
            darija_sw,
            options,
        )
    )

    output, dropped_metadata_rows = drop_rows_with_metadata_contamination(output, TARGET_COLUMNS)
    dropped_msa_darija_rows = 0
    if drop_msa_darija_leakage:
        output, dropped_msa_darija_rows = drop_rows_with_msa_darija_leakage(output)

    dropped_error_status_rows = 0
    if drop_error_status_rows:
        output, dropped_error_status_rows = drop_rows_by_status(output, {"ERROR"})

    top_words = discover_top_darija_words(output["darija_arabic"], darija_sw, top_n=100)
    cleaned, dropped_empty_rows = drop_rows_with_empty_targets(output, TARGET_COLUMNS)
    cleaned, post_clean_stats = apply_post_clean_quality_fixes(cleaned, TARGET_COLUMNS)
    if not keep_qc_columns:
        cleaned = drop_qc_columns(cleaned)

    stats = {
        "input_rows": int(len(df)),
        "dropped_rows": dropped_metadata_rows + dropped_msa_darija_rows + dropped_error_status_rows + dropped_empty_rows,
        "dropped_metadata_rows": dropped_metadata_rows,
        "dropped_msa_darija_rows": dropped_msa_darija_rows,
        "dropped_error_status_rows": dropped_error_status_rows,
        "dropped_empty_rows": dropped_empty_rows,
        "output_rows": int(len(cleaned)),
        **post_clean_stats,
    }
    return cleaned, top_words, stats


def parse_stopword_argument(raw_value: str | None, default_values: Iterable[str]) -> Set[str]:
    if not raw_value:
        return set(default_values)
    parsed = {tok.strip().lower() for tok in raw_value.split(",") if tok.strip()}
    return parsed


def create_dummy_dataset() -> pd.DataFrame:
    """Small in-script dataset to demonstrate pipeline execution."""
    return pd.DataFrame(
        {
            "darija_arabic": [
                "هدا @-@ مثال <unk> بسيط",
                "الْجُملَةُ الثّانِيَةُ فيها تَشْكِيلٌ",
                "<unk>",
            ],
            "darija_arabizi": [
                "hada @-@ mital 3la 2025 https://x.com",
                "kanmchi lmadrasa b7al 3ada!",
                "@-@",
            ],
            "english": [
                "This is @-@ a test <unk> row. Visit https://example.com",
                "Another SAMPLE, with punctuation!",
                "<unk>",
            ],
            "modern_standard_arabic": [
                "هٰذا @-@ نَصٌّ <unk> تَجْرِيبِيٌّ",
                "إِنَّ هٰذِهِ جُمْلَةٌ طَوِيلَةٌ",
                "@-@",
            ],
        }
    )


def print_dataframe_preview(title: str, df: pd.DataFrame, rows: int = 5) -> None:
    print(f"\n{title}")
    if df.empty:
        print("<empty dataframe>")
        return
    print(df.head(rows).to_string(index=False))


def run_dummy_demo(
    darija_stopwords: Iterable[str],
    arabizi_stopwords: Iterable[str],
    options: CleaningOptions,
) -> None:
    dummy_df = create_dummy_dataset()
    print_dataframe_preview("Dummy input DataFrame:", dummy_df, rows=10)

    cleaned_dummy, dummy_top_words, dummy_stats = clean_parallel_corpus(
        dummy_df,
        darija_stopwords=darija_stopwords,
        arabizi_stopwords=arabizi_stopwords,
        options=options,
    )

    print_dataframe_preview("Dummy cleaned DataFrame:", cleaned_dummy, rows=10)
    print(f"\nDummy stats: {json.dumps(dummy_stats, ensure_ascii=False)}")
    print("Top Darija words from dummy sample:")
    print(pd.DataFrame(dummy_top_words, columns=["word", "count"]).head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean corrected silver_shard_3 with modular NLP rules."
    )
    parser.add_argument(
        "--input",
        default="artifacts/silver_shard_3_qc/silver_shard_3.corrected.auto.csv",
        help="Path to corrected silver shard CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/silver_shard_3_cleaned",
        help="Isolated output folder for cleaned data and reports.",
    )
    parser.add_argument(
        "--output-name",
        default="silver_shard_3.corrected.cleaned.csv",
        help="Cleaned CSV file name.",
    )
    parser.add_argument(
        "--darija-stopwords",
        default=None,
        help="Comma-separated custom Darija Arabic stopwords.",
    )
    parser.add_argument(
        "--arabizi-stopwords",
        default=None,
        help="Comma-separated custom Darija Arabizi stopwords.",
    )
    parser.add_argument(
        "--skip-dummy-demo",
        action="store_true",
        help="Skip dummy DataFrame demonstration.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=8,
        help="Number of rows to print from final cleaned DataFrame.",
    )
    parser.add_argument(
        "--keep-qc-columns",
        action="store_true",
        help="Keep qc_changed_fields and qc_notes in the cleaned output CSV.",
    )
    parser.add_argument(
        "--keep-msa-darija-leakage",
        action="store_true",
        help="Keep rows where modern_standard_arabic appears to contain Darija markers.",
    )
    parser.add_argument(
        "--keep-error-status-rows",
        action="store_true",
        help="Keep rows where status is ERROR instead of dropping them.",
    )
    parser.add_argument(
        "--english-normalization",
        choices=("none", "stem", "lemma"),
        default="none",
        help="Apply optional English normalization: none, stemming, or lemmatization.",
    )
    parser.add_argument(
        "--disable-english-contraction-expansion",
        action="store_true",
        help="Disable expansion of common English contractions.",
    )
    parser.add_argument(
        "--split-arabic-clitics",
        action="store_true",
        help="Apply lightweight rule-based clitic segmentation for Arabic-script fields.",
    )
    parser.add_argument(
        "--split-darija-negation",
        action="store_true",
        help="When clitic splitting is enabled, split Moroccan negation pattern ma...sh.",
    )
    parser.add_argument(
        "--use-light-arabic-stemming",
        action="store_true",
        help="Apply lightweight Arabic prefix/suffix stemming for MSA and Darija Arabic.",
    )
    parser.add_argument(
        "--transliterate-arabizi-to-arabic",
        action="store_true",
        help="Transliterate Arabizi to Arabic script before cleaning.",
    )
    parser.add_argument(
        "--arabizi-drop-vowels",
        action="store_true",
        help="Drop Arabizi vowels (a,e,i,o,u) before token-level cleaning.",
    )
    parser.add_argument(
        "--arabizi-max-char-repeat",
        type=int,
        default=2,
        help="Maximum allowed repeated characters in Arabizi tokens.",
    )
    parser.add_argument(
        "--replace-emojis-with-token",
        action="store_true",
        help="Replace emojis with token 'emoji' instead of removing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    darija_stopwords = parse_stopword_argument(args.darija_stopwords, DEFAULT_DARIJA_STOPWORDS)
    arabizi_stopwords = parse_stopword_argument(args.arabizi_stopwords, DEFAULT_ARABIZI_STOPWORDS)
    options = CleaningOptions(
        english_normalization=args.english_normalization,
        expand_english_contractions=not args.disable_english_contraction_expansion,
        split_arabic_clitics=args.split_arabic_clitics,
        split_darija_negation=args.split_darija_negation,
        use_light_arabic_stemming=args.use_light_arabic_stemming,
        transliterate_arabizi_to_arabic=args.transliterate_arabizi_to_arabic,
        arabizi_drop_vowels=args.arabizi_drop_vowels,
        arabizi_max_char_repeat=max(1, int(args.arabizi_max_char_repeat)),
        emoji_replacement=" emoji " if args.replace_emojis_with_token else " ",
    )

    if not args.skip_dummy_demo:
        run_dummy_demo(darija_stopwords, arabizi_stopwords, options)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    cleaned_df, top_darija_words, stats = clean_parallel_corpus(
        df,
        darija_stopwords=darija_stopwords,
        arabizi_stopwords=arabizi_stopwords,
        options=options,
        keep_qc_columns=args.keep_qc_columns,
        drop_msa_darija_leakage=not args.keep_msa_darija_leakage,
        drop_error_status_rows=not args.keep_error_status_rows,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv_path = output_dir / args.output_name
    top_words_path = output_dir / "darija_arabic_top_100.csv"
    report_path = output_dir / "cleaning_report.json"

    cleaned_df.to_csv(cleaned_csv_path, index=False, encoding="utf-8")
    pd.DataFrame(top_darija_words, columns=["word", "count"]).to_csv(
        top_words_path,
        index=False,
        encoding="utf-8",
    )

    report = {
        "input_file": str(input_path),
        "cleaned_file": str(cleaned_csv_path),
        "top_darija_words_file": str(top_words_path),
        "kept_qc_columns": bool(args.keep_qc_columns),
        "kept_msa_darija_leakage": bool(args.keep_msa_darija_leakage),
        "kept_error_status_rows": bool(args.keep_error_status_rows),
        "english_normalization": options.english_normalization,
        "expand_english_contractions": options.expand_english_contractions,
        "split_arabic_clitics": options.split_arabic_clitics,
        "split_darija_negation": options.split_darija_negation,
        "use_light_arabic_stemming": options.use_light_arabic_stemming,
        "transliterate_arabizi_to_arabic": options.transliterate_arabizi_to_arabic,
        "arabizi_drop_vowels": options.arabizi_drop_vowels,
        "arabizi_max_char_repeat": options.arabizi_max_char_repeat,
        "emoji_replacement": "emoji" if args.replace_emojis_with_token else "removed",
        **stats,
    }
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    print_dataframe_preview("Final cleaned DataFrame preview:", cleaned_df, rows=args.preview_rows)
    print(f"\nCleaning report: {json.dumps(report, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
