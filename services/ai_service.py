
from __future__ import annotations

import base64
import json
import os
import re
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq
from PIL import Image, ImageOps

from .document_service import local_ocr_text, resize_image_for_vision
from .language_service import (
    chunk_text,
    clean_text,
    detect_language,
    env_int,
    env_str,
    infer_response_style,
    safe_json_loads,
)

try:
    from langdetect import detect as langdetect_detect
except Exception:
    langdetect_detect = None


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-./]*|[\u0900-\u0DFF][\u0900-\u0DFF\-./]*")
LATIN_RE = re.compile(r"[A-Za-z]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
GURMUKHI_RE = re.compile(r"[\u0A00-\u0A7F]")
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")

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


def image_to_data_url(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


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


def build_system_prompt(language: str, style: str = "concise") -> str:
    if style == "short":
        length_rule = """
- Give a short answer.
- Use 4-6 bullet points maximum.
- Never leave a sentence incomplete.
"""
    elif style == "detailed":
        length_rule = """
- Give a detailed answer.
- Explain causes, symptoms, precautions, tests, treatment, and diet if relevant.
- Use multiple sections with bullet points.
- Never leave a sentence incomplete.
"""
    else:
        length_rule = """
- Give a properly explained answer.
- Use enough detail for patient understanding.
- Prefer balanced, medium-to-detailed responses.
- Never leave a sentence incomplete.
"""

    return f"""
You are an advanced multilingual healthcare assistant.

PRIMARY GOAL:
- Help users clearly understand medical reports, symptoms, medicines, and health conditions.
- Explain medical information in simple patient-friendly language.

LANGUAGE RULES:
- Reply STRICTLY in the user's language.
- If the user asks in Marathi, answer fully in Marathi.
- If the user asks in Hindi, answer fully in Hindi.
- Do NOT switch back to English unless the user used English or asked for English.
- Do NOT mix Marathi/Hindi with English unnecessarily.
- Use natural sentences that sound like a real doctor explaining clearly.
- When a medicine or drug name must appear, keep it natural for the target language.
- On first mention of a medicine or drug, you may include the English original in parentheses.
- Never mix scripts inside one word.
- If the user explicitly asks for a language, honor it strictly.

MEDICAL RESPONSE RULES:
- Explain possible causes, symptoms, precautions, diet suggestions, tests, and treatment options when relevant.
- If lab report values are provided:
  - explain whether values are normal or abnormal
  - explain possible meaning
  - explain possible next tests
  - explain practical next steps
- Be careful and accurate. Do not invent values.

RESPONSE STYLE:
- Use clean bullet points when listing advice.
- Each bullet must be on a new line.
- Do NOT mix bullets with large paragraphs.
- Keep spacing clean and readable.

IMPORTANT:
- Do NOT say "I am not a doctor".
- Do NOT refuse simple medical questions.
- Do NOT overuse warnings.
- Give practical helpful explanations first.

{length_rule}

Language hint: {language}
""".strip()


def build_user_payload(
    user_message: str,
    file_context: str = "",
    image_context: str = "",
    image_ocr: str = "",
) -> str:
    parts: List[str] = []

    if user_message:
        parts.append(f"User message:\n{user_message.strip()}")

    if file_context.strip():
        parts.append(
            "Uploaded file context:\n"
            + file_context.strip()
            + "\n\nUse the uploaded document as grounding evidence."
        )

    if image_context.strip():
        parts.append(
            "Uploaded image context:\n"
            + image_context.strip()
            + "\n\nUse the image context carefully and mention uncertainty if needed."
        )

    if image_ocr.strip():
        parts.append("OCR text from image:\n" + image_ocr.strip())

    parts.append(
        "Answer safely. If medical advice is uncertain, say so and suggest a clinician when appropriate."
    )
    return "\n\n".join(parts)


def build_chat_messages(
    user_message: str,
    language: str,
    recent_messages: List[Dict[str, str]],
    memory_summary: str = "",
    file_context: str = "",
    image_context: str = "",
    image_ocr: str = "",
) -> List[Dict[str, Any]]:
    style = infer_response_style(user_message)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": f"TARGET_LANGUAGE_CODE={language}"},
        {"role": "system", "content": build_system_prompt(language, style)},
    ]

    if memory_summary.strip():
        messages.append(
            {
                "role": "system",
                "content": f"Conversation memory summary:\n{memory_summary.strip()}",
            }
        )

    for msg in recent_messages:
        role = msg.get("role", "")
        content = clean_text(msg.get("content", ""))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": build_user_payload(
                user_message=user_message,
                file_context=file_context,
                image_context=image_context,
                image_ocr=image_ocr,
            ),
        }
    )
    return messages


@dataclass
class ImageInsight:
    filename: str = ""
    description: str = ""
    visible_text: str = ""
    medical_relevance: str = ""
    objects: List[str] = field(default_factory=list)
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_context_text(self) -> str:
        objects_text = ", ".join(self.objects) if self.objects else "None"
        return clean_text(
            f"Filename: {self.filename}\n"
            f"Description: {self.description or 'Not available'}\n"
            f"Objects: {objects_text}\n"
            f"Medical relevance: {self.medical_relevance or 'Not available'}\n"
            f"Visible text: {self.visible_text or 'None'}\n"
            f"Notes: {self.notes or 'None'}"
        )

    def to_short_response(self) -> str:
        parts: List[str] = []
        if self.description:
            parts.append(self.description)
        if self.visible_text:
            parts.append(f"Text seen: {self.visible_text}")
        if self.medical_relevance:
            parts.append(f"Medical note: {self.medical_relevance}")
        return clean_text(" ".join(parts)) or "Image processed successfully."


class MedicalTermNormalizer:
    def __init__(self, groq_service: "GroqService") -> None:
        self.groq = groq_service

    def _is_medical_candidate(self, token: str, context: str) -> bool:
        t = clean_text(token)
        if not t:
            return False

        low = t.lower().strip(".,;:!?()[]{}\"'`")
        if not low or low in EN_STOPWORDS:
            return False

        has_latin = bool(LATIN_RE.search(t))
        has_indic = bool(
            DEVANAGARI_RE.search(t)
            or BENGALI_RE.search(t)
            or GURMUKHI_RE.search(t)
            or GUJARATI_RE.search(t)
            or TAMIL_RE.search(t)
            or TELUGU_RE.search(t)
            or KANNADA_RE.search(t)
            or MALAYALAM_RE.search(t)
        )
        mixed = has_latin and has_indic

        if mixed:
            return True

        if any(low.endswith(suf) for suf in MEDICATION_SUFFIXES):
            return True

        if has_latin and len(low) >= 4:
            if any(cw in context.lower() for cw in COMMON_CONTEXT_WORDS):
                return True
            if low[0].isupper():
                return True
            if re.search(r"\d", low) or "-" in low:
                return True

        if len(low) >= 5 and (low in context.lower()):
            if any(cw in context.lower() for cw in COMMON_CONTEXT_WORDS):
                return True

        return False

    def extract_candidates(self, answer: str, source_text: str = "") -> List[str]:
        text = f"{answer or ''} {source_text or ''}".strip()
        if not text:
            return []

        candidates: List[str] = []
        seen: set[str] = set()

        for match in TOKEN_RE.finditer(text):
            token = match.group(0)
            if not self._is_medical_candidate(token, text):
                continue

            normalized = token.strip(".,;:!?()[]{}\"'`")
            if not normalized:
                continue

            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(normalized)

        for m in re.finditer(r"[\u0900-\u0DFF]+[A-Za-z][A-Za-z0-9\-./]*", text):
            token = m.group(0).strip(".,;:!?()[]{}\"'`")
            key = token.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(token)

        return candidates[:20]

    def _needs_normalization(self, answer: str, target_language: str, candidates: List[str]) -> bool:
        if not answer or target_language.lower().startswith("en"):
            return False
        if candidates:
            return True
        if LATIN_RE.search(answer):
            return True
        return False

    def _rewrite_candidates_via_llm(
        self,
        answer: str,
        target_language: str,
        candidates: List[str],
        source_text: str = "",
    ) -> Dict[str, str]:
        if not self.groq.client:
            return {}

        prompt = f"""
You are a multilingual medical text normalizer.

Task:
- The response must be fully in {target_language}.
- Detect the following medical terms / drug names / mixed-script words.
- Return the best localized rendering for each term.
- For medicine/drug names, format as: Local-language-form (English original)
- Never return mixed-script words like "परacetamol".
- Keep meaning unchanged.
- Do not add new facts.
- Do not explain anything.

Return ONLY valid JSON in this exact format:
{{"terms": {{"original_term":"localized term (Original Term)"}}}}

Original response:
{answer}

Source context:
{source_text}

Terms to normalize:
{json.dumps(candidates, ensure_ascii=False)}
""".strip()

        try:
            raw = self.groq.chat(
                [
                    {"role": "system", "content": "You normalize medical text and return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=280,
                retries=1,
                response_format={"type": "json_object"},
                postprocess=False,
            )
        except Exception:
            return {}

        data = safe_json_loads(raw)
        terms = data.get("terms") if isinstance(data, dict) else None
        if not isinstance(terms, dict):
            return {}

        normalized: Dict[str, str] = {}
        for k, v in terms.items():
            k = clean_text(str(k))
            v = clean_text(str(v))
            if k and v:
                normalized[k] = v
        return normalized

    def normalize_answer(self, answer: str, target_language: str, source_text: str = "") -> str:
        answer = clean_text(answer)
        if not answer:
            return answer

        candidates = self.extract_candidates(answer, source_text=source_text)
        if not self._needs_normalization(answer, target_language, candidates):
            return self._final_polish(answer, target_language)

        replacements = self._rewrite_candidates_via_llm(
            answer,
            target_language=target_language,
            candidates=candidates,
            source_text=source_text,
        )

        for orig, repl in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            if not orig or not repl:
                continue
            pattern = re.compile(rf"\b{re.escape(orig)}\b", re.IGNORECASE)
            answer = pattern.sub(repl, answer)

        answer = self._final_polish(answer, target_language)
        return answer

    def _final_polish(self, answer: str, target_language: str) -> str:
        answer = clean_text(answer)
        answer = re.sub(r"\(\s+", "(", answer)
        answer = re.sub(r"\s+\)", ")", answer)
        answer = re.sub(r"([\u0900-\u0DFF]+)([A-Za-z]+)", r"\1 \2", answer)

        if target_language.startswith("mr") or target_language == "mixed":
            replacements = {
                "paracetamol": "पॅरासिटामोल (Paracetamol)",
                "crocin": "क्रोसिन (Crocin)",
                "dolo": "डोलो (Dolo)",
                "ibuprofen": "आयबुप्रोफेन (Ibuprofen)",
            }
            for eng, mar in replacements.items():
                pattern = re.compile(rf"\b{re.escape(eng)}\b", re.IGNORECASE)
                answer = pattern.sub(mar, answer)

        return answer


class GroqService:
    def __init__(self) -> None:
        self.api_key = env_str("GROQ_API_KEY", "")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.text_model = env_str("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
        self.vision_model = env_str("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.stt_model = env_str("GROQ_STT_MODEL", "whisper-large-v3-turbo")
        self.file_summary_threshold_chars = env_int("FILE_SUMMARY_THRESHOLD_CHARS", 12000)

        tesseract_cmd = env_str("TESSERACT_CMD", "")
        if tesseract_cmd:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            except Exception:
                pass

        self.normalizer = MedicalTermNormalizer(self)

    def _require_client(self) -> Groq:
        if self.client is None:
            raise RuntimeError("GROQ_API_KEY is missing. Please add it to your .env file.")
        return self.client

    def _extract_last_user_text(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return clean_text(content)
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(str(item.get("text", "")))
                    return clean_text(" ".join(parts))
        return ""

    def _infer_style_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        return infer_response_style(self._extract_last_user_text(messages))

    def _infer_target_language_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        for msg in messages or []:
            if msg.get("role") != "system":
                continue
            content = clean_text(msg.get("content", ""))
            m = re.search(r"TARGET_LANGUAGE_CODE\s*=\s*([a-zA-Z\-]+)", content)
            if m:
                return m.group(1).strip().lower()
            m2 = re.search(r"Language hint:\s*([a-zA-Z\-]+)", content)
            if m2:
                return m2.group(1).strip().lower()
        return "en"

    def _token_cap(self, style: str) -> int:
        if style == "short":
            return 500
        if style == "detailed":
            return 2200
        return 1400

    def _chat_raw(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.25,
        max_tokens: int = 900,
        response_format: Optional[Dict[str, Any]] = None,
        retries: int = 2,
    ) -> str:
        client = self._require_client()
        style = self._infer_style_from_messages(messages)
        bounded_tokens = min(max_tokens or 900, self._token_cap(style))

        kwargs: Dict[str, Any] = {
            "model": model or self.text_model,
            "messages": messages,
            "temperature": 0.2 if style != "detailed" else temperature,
            "top_p": 1,
            "max_completion_tokens": bounded_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content if resp.choices else ""
                return clean_text(content or "")
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.7 * (attempt + 1))
                else:
                    break

        raise RuntimeError(clean_text(str(last_error)) or "Connection error.")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.25,
        max_tokens: int = 900,
        response_format: Optional[Dict[str, Any]] = None,
        retries: int = 2,
        postprocess: bool = True,
        source_text: str = "",
    ) -> str:
        content = self._chat_raw(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            retries=retries,
        )

        if not postprocess:
            return content

        target_language = self._infer_target_language_from_messages(messages)
        source = source_text or self._extract_last_user_text(messages)
        return self.normalizer.normalize_answer(content, target_language, source_text=source)

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.25,
        max_tokens: int = 900,
        retries: int = 2,
    ):
        client = self._require_client()
        style = self._infer_style_from_messages(messages)
        bounded_tokens = min(max_tokens or 900, self._token_cap(style))

        kwargs: Dict[str, Any] = {
            "model": model or self.text_model,
            "messages": messages,
            "temperature": 0.2 if style != "detailed" else temperature,
            "top_p": 1,
            "max_completion_tokens": bounded_tokens,
            "stream": True,
        }

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                stream = client.chat.completions.create(**kwargs)
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta.content or ""
                    except Exception:
                        delta = ""
                    if delta:
                        yield delta
                return
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.7 * (attempt + 1))
                else:
                    break

        raise RuntimeError(clean_text(str(last_error)) or "Connection error.")

    def summarize_text(self, text: str, language: str = "en", max_tokens: int = 350) -> str:
        text = clean_text(text)
        if not text:
            return ""

        prompt = f"""
Summarize this text for a healthcare chatbot.

Keep only:
- the main facts
- symptoms, dates, medications, numeric values, and risks
- any follow-up questions that should be asked

Write in the same language as the user's context: {language}

TEXT:
{text}
""".strip()

        return self.chat(
            [
                {"role": "system", "content": "You summarize content accurately and concisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=max_tokens,
            retries=1,
            postprocess=False,
        )

    def condense_document_context(self, text: str, language: str = "en") -> str:
        text = clean_text(text)
        if not text:
            return ""
        if len(text) <= self.file_summary_threshold_chars:
            return text
        return self.summarize_large_text(text, language=language)

    def summarize_large_text(self, text: str, language: str = "en") -> str:
        text = clean_text(text)
        if not text:
            return ""
        if len(text) <= self.file_summary_threshold_chars:
            return text

        chunks = chunk_text(text, chunk_size=7000, overlap=500)
        if not chunks:
            return ""

        partials: List[str] = []
        for chunk in chunks[:5]:
            try:
                partials.append(self.summarize_text(chunk, language=language, max_tokens=280))
            except Exception:
                partials.append(safe_snippet(chunk, 2200))

        merged = "\n\n".join(partials)
        if len(chunks) > 5:
            merged += "\n\n[More content was condensed because the file was large.]"

        try:
            final = self.summarize_text(merged, language=language, max_tokens=280)
            return final or merged
        except Exception:
            return merged

    def compress_chat_memory(self, previous_summary: str, older_messages: str, language: str = "en") -> str:
        previous_summary = clean_text(previous_summary)
        older_messages = clean_text(older_messages)

        prompt = f"""
You are compressing conversation memory for a healthcare chatbot.

Keep only useful long-term memory:
- symptoms and medical context
- user preferences
- ongoing questions
- important conclusions or warnings
- anything needed to continue the same conversation naturally

Do NOT add new medical advice.
Do NOT hallucinate facts.
Be concise.

Existing summary:
{previous_summary or "(none)"}

Older messages:
{older_messages}

Return a compact memory summary in the same language as the conversation context: {language}
""".strip()

        return self.chat(
            [
                {"role": "system", "content": "You compress conversation memory safely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=280,
            retries=1,
            postprocess=False,
        )

    def generate_followups(
        self,
        *,
        user_message: str,
        assistant_answer: str,
        language: str = "en",
        memory_summary: str = "",
        file_context: str = "",
        image_context: str = "",
        image_ocr: str = "",
    ) -> List[str]:
        prompt = f"""
Generate 2 to 4 short follow-up questions for a healthcare chatbot.

Rules:
- Questions must be directly relevant to the last answer and user context.
- Keep them short, natural, and useful.
- Use the same language as the conversation: {language}
- Return ONLY valid JSON in this exact format:
  {{"questions":["question 1","question 2"]}}

Context:
User message: {user_message}
Assistant answer: {assistant_answer}
Memory summary: {memory_summary}
File context: {file_context}
Image context: {image_context}
OCR text: {image_ocr}
""".strip()

        raw = self.chat(
            [
                {"role": "system", "content": "You generate concise follow-up questions as JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
            retries=1,
            response_format={"type": "json_object"},
            postprocess=False,
        )

        data = safe_json_loads(raw)
        questions = data.get("questions") if isinstance(data, dict) else None
        if not isinstance(questions, list):
            questions = []

        cleaned: List[str] = []
        for q in questions:
            q = clean_text(str(q))
            if not q:
                continue
            if not q.endswith("?"):
                q += "?"
            cleaned.append(q)

        return cleaned[:4]

    def analyze_image(
        self,
        image_path: str,
        *,
        user_question: str = "",
        language: str = "en",
    ) -> ImageInsight:
        client = self._require_client()

        max_side = env_int("IMAGE_MAX_SIDE", 1600)
        prepared_path = resize_image_for_vision(image_path, max_side=max_side)
        ocr_text = local_ocr_text(prepared_path)

        base_prompt = f"""
Analyze the image for a healthcare chatbot.

What to return:
- description of the scene or objects
- visible text if any
- medical relevance if any
- possible concerns
- concise notes for later reasoning

User question: {user_question or "No specific question"}
Language context: {language}

If the image is not medical, still describe it clearly.
If you are unsure, say so.
Return valid JSON with keys:
description, visible_text, medical_relevance, objects, notes
""".strip()

        if ocr_text:
            base_prompt += f"\n\nLocal OCR text already extracted:\n{ocr_text}"

        parsed: Dict[str, Any] = {}
        raw_text = ""

        try:
            data_url = image_to_data_url(prepared_path)
            for use_json in (True, False):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": self.vision_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": base_prompt},
                                    {"type": "image_url", "image_url": {"url": data_url}},
                                ],
                            }
                        ],
                        "temperature": 0.15,
                        "top_p": 1,
                        "max_completion_tokens": 450,
                    }
                    if use_json:
                        kwargs["response_format"] = {"type": "json_object"}

                    resp = client.chat.completions.create(**kwargs)
                    raw_text = resp.choices[0].message.content or ""
                    parsed = safe_json_loads(raw_text)
                    if parsed:
                        break
                except Exception:
                    continue
        finally:
            try:
                os.remove(prepared_path)
            except Exception:
                pass

        description = clean_text(str(parsed.get("description", "") if parsed else ""))
        visible_text = clean_text(str(parsed.get("visible_text", "") if parsed else ""))
        medical_relevance = clean_text(str(parsed.get("medical_relevance", "") if parsed else ""))
        notes = clean_text(str(parsed.get("notes", "") if parsed else ""))

        objects = parsed.get("objects", []) if isinstance(parsed, dict) else []
        if not isinstance(objects, list):
            objects = []

        if not description and raw_text and not parsed:
            description = clean_text(raw_text[:1200])

        if not description and not visible_text and not medical_relevance:
            description = "Image analysis completed with limited details available."
            if not visible_text:
                visible_text = ocr_text or ""
            if not notes:
                notes = "Fallback analysis was used because the vision response could not be fully parsed."

        if not visible_text and ocr_text:
            visible_text = ocr_text

        return ImageInsight(
            filename=Path(image_path).name,
            description=description,
            visible_text=visible_text,
            medical_relevance=medical_relevance,
            objects=[clean_text(str(x)) for x in objects if clean_text(str(x))],
            notes=notes,
            raw=parsed or {"raw": raw_text},
        )

    def transcribe_audio_file(self, audio_path: str, language: Optional[str] = None) -> str:
        client = self._require_client()

        with open(audio_path, "rb") as f:
            kwargs: Dict[str, Any] = {
                "file": f,
                "model": self.stt_model,
                "temperature": 0.0,
            }
            if language:
                kwargs["language"] = language

            transcription = client.audio.transcriptions.create(**kwargs)

        if hasattr(transcription, "text"):
            return clean_text(transcription.text)

        return clean_text(str(transcription))
