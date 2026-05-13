
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

try:
    from langdetect import detect as langdetect_detect
except Exception:
    langdetect_detect = None


WHITESPACE_RE = re.compile(r"\s+")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
GURMUKHI_RE = re.compile(r"[\u0A00-\u0A7F]")
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")
LATIN_RE = re.compile(r"[A-Za-z]")

HI_HINTS = {
    "hai", "kya", "kyu", "kyun", "dard", "bukhar", "khansi", "sardi",
    "pet", "sar", "doctor", "dawai", "kripya", "please", "nahi", "ho",
}
MR_HINTS = {
    "आहे", "काय", "का", "दुखत", "ताप", "खोकला", "डॉक्टर", "औषध",
    "कृपया", "होत", "नाही", "मला", "सांग", "सांगू",
}

EMERGENCY_KEYWORDS = {
    "chest pain", "difficulty breathing", "shortness of breath",
    "seizure", "stroke", "unconscious", "fainting", "blue lips",
    "severe bleeding", "suicidal", "self harm", "overdose",
    "heart attack", "anaphylaxis", "confusion", "one-sided weakness",
    "sudden vision loss", "severe allergic reaction",
    "छाती दुख", "श्वास घेण्यास त्रास", "बेशुद्ध", "अचानक कमजोरी",
    "आत्महत्या", "स्वतःला इजा", "तीव्र रक्तस्राव",
    "सीने में दर्द", "सांस लेने में दिक्कत", "बेहोशी",
}

COMMON_CONTEXT_WORDS = {
    "tablet", "tab", "capsule", "caps", "syrup", "dose", "dosage", "mg", "ml", "injection",
    "medicine", "medication", "drug", "antibiotic", "antibiotics", "painkiller", "analgesic",
    "fever", "cold", "cough", "pain", "headache", "stomach", "nausea", "vomiting", "diarrhea",
    "allergy", "allergic", "prescribed", "prescription", "take", "taken", "take it", "use",
    "should", "avoid", "side effect", "side effects", "doctor", "hospital", "clinic",
    "ताप", "औषध", "गोळी", "सिरप", "डोस", "इंजेक्शन", "वेदना", "डोकेदुखी",
    "fever", "bukhar", "khansi", "sardi", "दवाई", "डोके", "पोट",
}

MEDICATION_SUFFIXES = (
    "mab", "nib", "vir", "cillin", "mycin", "azole", "statin", "olol", "pril", "sartan",
    "prazole", "dine", "azine", "oxetine", "caine", "sone", "lone", "tidine", "fen",
    "acetamol", "olam", "pam", "lam", "peridone", "triptyline", "barbital", "cort", "pred",
)

EN_STOPWORDS = {
    "the", "and", "or", "but", "if", "then", "also", "with", "without", "for", "to", "in", "on",
    "at", "by", "from", "of", "a", "an", "is", "are", "was", "were", "be", "been", "it", "this",
    "that", "these", "those", "you", "your", "we", "they", "he", "she", "them", "his", "her",
    "my", "me", "our", "their", "as", "not", "no", "do", "does", "did", "can", "could", "will",
    "would", "should", "may", "might", "must", "have", "has", "had", "take", "taken", "use",
    "used", "about", "because", "when", "what", "which", "who", "whom", "where", "why", "how",
    "please", "thanks", "thank", "okay", "ok",
}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_str(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return (value or "").strip() or default


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def safe_snippet(text: str, limit: int = 12000) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ...[truncated]"


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = clean_text(text)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except Exception:
        pass

    match = JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}


def chunk_text(text: str, chunk_size: int = 7000, overlap: int = 500) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + chunk_size, total)
        chunks.append(text[start:end])
        if end >= total:
            break
        start = max(0, end - overlap)
    return chunks


def detect_language(text: str) -> str:
    text = clean_text(text)
    if not text:
        return "en"

    lower = text.lower()
    hi_score = sum(lower.count(tok) for tok in HI_HINTS)
    mr_score = sum(text.count(tok) for tok in MR_HINTS)

    if langdetect_detect and len(text) >= 20:
        try:
            lang = langdetect_detect(text)
            if lang.startswith("hi"):
                return "hi"
            if lang.startswith("mr"):
                return "mr"
            if lang.startswith("en"):
                return "en"
        except Exception:
            pass

    has_dev = bool(DEVANAGARI_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))

    if has_dev and has_latin:
        return "mixed"
    if has_dev:
        return "mr" if mr_score > hi_score else "hi"

    if hi_score > 0 and mr_score > 0:
        return "mixed"
    if mr_score > hi_score and mr_score > 0:
        return "mr"
    if hi_score > 0:
        return "hi"

    return "en"


def is_emergency_text(text: str) -> bool:
    text = clean_text(text).lower()
    if not text:
        return False
    return any(keyword in text for keyword in EMERGENCY_KEYWORDS)


def infer_response_style(user_text: str) -> str:
    text = clean_text(user_text).lower()

    short_patterns = [
        r"\bshort\b",
        r"\bbrief\b",
        r"\bconcise\b",
        r"\bin 3 lines\b",
        r"\b3 lines\b",
        r"\bthree lines\b",
        r"\b2 lines\b",
        r"\bone line\b",
        r"\bquick\b",
        r"\bvery short\b",
    ]
    long_patterns = [
        r"\bdetailed\b",
        r"\blong\b",
        r"\bvery long\b",
        r"\bexplain in detail\b",
        r"\bdeep explanation\b",
        r"\bmore detail\b",
        r"\bin depth\b",
    ]

    if any(re.search(p, text) for p in short_patterns):
        return "short"
    if any(re.search(p, text) for p in long_patterns):
        return "detailed"
    return "concise"


def _lang_to_script(lang: str) -> str:
    lang = (lang or "").lower()
    if lang.startswith(("hi", "mr")):
        return "devanagari"
    if lang.startswith("kn"):
        return "kannada"
    if lang.startswith("ta"):
        return "tamil"
    if lang.startswith("te"):
        return "telugu"
    if lang.startswith("ml"):
        return "malayalam"
    if lang.startswith("bn"):
        return "bengali"
    if lang.startswith("gu"):
        return "gujarati"
    if lang.startswith("pa"):
        return "gurmukhi"
    return "latin"
