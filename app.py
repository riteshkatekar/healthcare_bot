#app.py

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context
from werkzeug.utils import secure_filename

from services import (
    ALLOWED_AUDIO_EXTS,
    ALLOWED_IMAGE_EXTS,
    ALLOWED_TEXT_EXTS,
    GroqService,
    ImageInsight,
    MemoryStore,
    build_chat_messages,
    clean_text,
    detect_language,
    extract_text_from_file,
    is_emergency_text,
)

load_dotenv()

# -------------------------------------------------------------------
# Paths / config
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
FILES_DIR = UPLOAD_DIR / "files"
IMAGES_DIR = UPLOAD_DIR / "images"
AUDIO_DIR = UPLOAD_DIR / "audio"
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "chatbot.sqlite3")))

for path in (UPLOAD_DIR, FILES_DIR, IMAGES_DIR, AUDIO_DIR, DB_PATH.parent):
    path.mkdir(parents=True, exist_ok=True)

MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "25"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
SUMMARY_TRIGGER_MESSAGES = int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "28"))
UPLOAD_TTL_SECONDS = int(os.getenv("UPLOAD_TTL_SECONDS", "1800"))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "600"))
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").strip().lower() == "true"

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = SESSION_COOKIE_SECURE

groq_service = GroqService()
memory_store = MemoryStore(str(DB_PATH), max_history_messages=MAX_HISTORY_MESSAGES)

_cleanup_lock = threading.Lock()
_cleanup_started = False


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def sse_data(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def get_session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex
    return session["session_id"]


def request_payload() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    json_payload = request.get_json(silent=True)
    if isinstance(json_payload, dict):
        data.update(json_payload)
    if request.form:
        data.update(request.form.to_dict(flat=True))
    return data


def get_payload_value(payload: Dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def payload_bool(payload: Dict[str, Any], key: str) -> bool:
    value = str(payload.get(key, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def normalize_message(text: str) -> str:
    return clean_text(text or "")


def save_upload(file_storage, target_dir: Path) -> str:
    target_dir.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(file_storage.filename or "")
    if not original_name:
        raise ValueError("Invalid upload filename.")
    path = target_dir / f"{uuid.uuid4().hex}_{original_name}"
    file_storage.save(str(path))
    return str(path)


def cleanup_old_uploads() -> None:
    now = time.time()

    if not UPLOAD_DIR.exists():
        return

    for path in UPLOAD_DIR.rglob("*"):
      try:
          if not path.is_file():
              continue
          if path.name.startswith("."):
              continue
          age = now - path.stat().st_mtime
          if age > UPLOAD_TTL_SECONDS:
              path.unlink(missing_ok=True)
      except Exception:
          continue

    # Remove empty directories bottom-up.
    for folder in sorted([p for p in UPLOAD_DIR.rglob("*") if p.is_dir()], key=lambda p: len(str(p)), reverse=True):
        try:
            next(folder.iterdir())
        except StopIteration:
            try:
                folder.rmdir()
            except Exception:
                pass
        except Exception:
            pass


def _cleanup_loop() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        cleanup_old_uploads()


def start_cleanup_thread() -> None:
    global _cleanup_started
    with _cleanup_lock:
        if _cleanup_started:
            return
        _cleanup_started = True
        thread = threading.Thread(target=_cleanup_loop, name="upload-cleanup", daemon=True)
        thread.start()


@app.before_request
def ensure_background_thread_started() -> None:
    start_cleanup_thread()


def normalize_stream_chunk(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, (int, float)):
        return str(chunk)

    # Common dict-like payloads.
    if isinstance(chunk, dict):
        for key in ("text", "content", "delta", "token"):
            value = chunk.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                nested = value.get("text") or value.get("content")
                if isinstance(nested, str):
                    return nested
        return ""

    # Common object-like payloads.
    for attr in ("text", "content", "delta"):
        value = getattr(chunk, attr, None)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = value.get("text") or value.get("content")
            if isinstance(nested, str):
                return nested

    return ""


def get_first_image_file(req):
    image_files = req.files.getlist("images")
    image_files = [f for f in image_files if f and getattr(f, "filename", "")]
    if image_files:
        return image_files[0]

    single = req.files.get("image")
    if single and single.filename:
        return single

    return None


def get_all_image_files(req) -> List[Any]:
    files = req.files.getlist("images") or []
    files = [f for f in files if f and getattr(f, "filename", "")]
    if files:
        return files

    single = req.files.get("image")
    if single and single.filename:
        return [single]

    return []


def attachment_signature(attachments: List[Dict[str, Any]]) -> str:
    if not attachments:
        return ""
    parts: List[str] = []
    for item in attachments:
        parts.append(
            f"{item.get('type','')}:"
            f"{item.get('filename','')}:"
            f"{item.get('file_type','')}:"
            f"{item.get('status','')}"
        )
    return "|".join(parts)


# -------------------------------------------------------------------
# Context preparation
# -------------------------------------------------------------------
def prepare_file_context(file_storage, user_message: str) -> Tuple[str, Dict[str, Any]]:
    suffix = Path(file_storage.filename).suffix.lower()
    if suffix not in ALLOWED_TEXT_EXTS:
        raise ValueError(f"Unsupported file type: {suffix}")

    path = save_upload(file_storage, FILES_DIR)
    extracted = extract_text_from_file(path, file_storage.filename)

    attachment = {
        "type": "file",
        "filename": file_storage.filename,
        "status": "uploaded",
        "file_type": suffix.lstrip("."),
        "path": Path(path).name,
    }

    if not extracted:
        attachment["status"] = "no_text_extracted"
        attachment["summary"] = ""
        return "", attachment

    language_hint = detect_language(f"{user_message} {extracted}")
    condensed = groq_service.condense_document_context(extracted, language=language_hint)
    condensed = condensed or extracted[:12000]

    attachment["status"] = "processed"
    attachment["summary"] = condensed[:500] if condensed else ""
    return condensed, attachment


def prepare_image_context(image_files, user_message: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    if not image_files:
        return "", "", []

    language_hint = detect_language(user_message)
    context_blocks: List[str] = []
    ocr_blocks: List[str] = []
    attachments: List[Dict[str, Any]] = []

    for idx, file_storage in enumerate(image_files, start=1):
        suffix = Path(file_storage.filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTS:
            raise ValueError(f"Unsupported image type: {suffix}")

        path = save_upload(file_storage, IMAGES_DIR)
        insight: ImageInsight = groq_service.analyze_image(
            path,
            user_question=user_message,
            language=language_hint,
        )

        context_blocks.append(f"Image {idx} ({file_storage.filename}):\n{insight.to_context_text()}")
        if insight.visible_text:
            ocr_blocks.append(insight.visible_text)

        attachments.append(
            {
                "type": "image",
                "filename": file_storage.filename,
                "status": "processed",
                "file_type": suffix.lstrip("."),
                "description": insight.description,
                "visible_text": insight.visible_text,
                "medical_relevance": insight.medical_relevance,
                "objects": insight.objects,
                "summary": insight.to_short_response(),
                "path": Path(path).name,
            }
        )

    combined_context = "\n\n".join(context_blocks).strip()
    combined_ocr = "\n".join(ocr_blocks).strip()
    return combined_context, combined_ocr, attachments


def build_generation_context(
    session_id: str,
    user_message: str,
    *,
    file_context: str = "",
    image_context: str = "",
    image_ocr: str = "",
    exclude_last_assistant: bool = False,
):
    summary, recent_messages = memory_store.get_context(session_id, keep_last=MAX_HISTORY_MESSAGES)

    if exclude_last_assistant and recent_messages and recent_messages[-1].get("role") == "assistant":
        recent_messages = recent_messages[:-1]

    language_hint = detect_language(
        " ".join(
            part
            for part in [user_message, file_context, image_context, image_ocr, summary]
            if part
        )
    )

    messages = build_chat_messages(
        user_message=user_message,
        language=language_hint,
        recent_messages=recent_messages,
        memory_summary=summary,
        file_context=file_context,
        image_context=image_context,
        image_ocr=image_ocr,
    )

    return summary, language_hint, messages


def finalize_turn(
    session_id: str,
    user_message: str,
    answer: str,
    *,
    language_hint: str,
    file_context: str = "",
    image_context: str = "",
    image_ocr: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    endpoint: str = "chat",
    record_history: bool = True,
    replace_last_assistant: bool = False,
) -> None:
    if record_history:
        memory_store.add_message(session_id, "user", user_message)
        memory_store.add_message(session_id, "assistant", answer)

        try:
            memory_store.compact_if_needed(
                session_id,
                groq_service,
                language=language_hint,
                keep_last=MAX_HISTORY_MESSAGES,
                trigger_after=SUMMARY_TRIGGER_MESSAGES,
            )
        except Exception:
            pass
    elif replace_last_assistant:
        memory_store.replace_last_assistant(session_id, answer)

    memory_store.set_turn_cache(
        session_id,
        user_message=user_message,
        assistant_answer=answer,
        file_context=file_context,
        image_context=image_context,
        image_ocr=image_ocr,
        attachments=attachments or [],
        endpoint=endpoint,
    )


def make_response_payload(
    *,
    session_id: str,
    user_message: str,
    answer: str,
    language_hint: str,
    attachments: List[Dict[str, Any]],
    file_context: str,
    image_context: str,
    image_ocr: str,
    endpoint: str,
) -> Dict[str, Any]:
    emergency = any(
        is_emergency_text(part)
        for part in [user_message, answer, file_context, image_context, image_ocr]
        if part
    )

    followups: List[str] = []
    try:
        summary, _, _ = build_generation_context(
            session_id,
            user_message,
            file_context=file_context,
            image_context=image_context,
            image_ocr=image_ocr,
        )
        followups = groq_service.generate_followups(
            user_message=user_message,
            assistant_answer=answer,
            language=language_hint,
            memory_summary=summary,
            file_context=file_context,
            image_context=image_context,
            image_ocr=image_ocr,
        )
    except Exception:
        followups = []

    return {
        "ok": True,
        "session_id": session_id,
        "answer": answer,
        "reply": answer,
        "language": language_hint,
        "emergency": emergency,
        "followups": followups,
        "attachments": attachments,
        "file_used": bool(file_context),
        "image_used": bool(image_context or image_ocr),
        "endpoint": endpoint,
    }


def compose_answer(
    session_id: str,
    user_message: str,
    *,
    file_context: str = "",
    image_context: str = "",
    image_ocr: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    endpoint: str = "chat",
    record_history: bool = True,
    replace_last_assistant: bool = False,
    exclude_last_assistant: bool = False,
) -> Dict[str, Any]:
    user_message = normalize_message(user_message)
    if not user_message:
        user_message = "Please analyze the uploaded content and answer based on it."

    summary, language_hint, messages = build_generation_context(
        session_id,
        user_message,
        file_context=file_context,
        image_context=image_context,
        image_ocr=image_ocr,
        exclude_last_assistant=exclude_last_assistant,
    )

    try:
        answer = groq_service.chat(messages, temperature=0.3, max_tokens=900)
    except Exception as exc:
        answer = (
            "I’m sorry, I could not generate a response right now. "
            f"Technical detail: {clean_text(str(exc))}"
        )

    if not answer:
        answer = "I’m sorry, I could not generate a response right now."

    finalize_turn(
        session_id,
        user_message,
        answer,
        language_hint=language_hint,
        file_context=file_context,
        image_context=image_context,
        image_ocr=image_ocr,
        attachments=attachments,
        endpoint=endpoint,
        record_history=record_history,
        replace_last_assistant=replace_last_assistant,
    )

    return make_response_payload(
        session_id=session_id,
        user_message=user_message,
        answer=answer,
        language_hint=language_hint,
        attachments=attachments or [],
        file_context=file_context,
        image_context=image_context,
        image_ocr=image_ocr,
        endpoint=endpoint,
    )


def stream_answer(
    *,
    session_id: str,
    user_message: str,
    file_context: str = "",
    image_context: str = "",
    image_ocr: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    endpoint: str = "chat",
    record_history: bool = True,
    replace_last_assistant: bool = False,
    exclude_last_assistant: bool = False,
):
    user_message = normalize_message(user_message)
    if not user_message:
        user_message = "Please analyze the uploaded content and answer based on it."

    summary, language_hint, messages = build_generation_context(
        session_id,
        user_message,
        file_context=file_context,
        image_context=image_context,
        image_ocr=image_ocr,
        exclude_last_assistant=exclude_last_assistant,
    )

    def generate():
        answer_parts: List[str] = []
        yield sse_data({"type": "meta", "language": language_hint})

        try:
            for chunk in groq_service.stream_chat(messages, temperature=0.3, max_tokens=900):
                token = normalize_stream_chunk(chunk)
                if not token:
                    continue
                answer_parts.append(token)
                yield sse_data({"type": "token", "text": token})

            answer = clean_text("".join(answer_parts))
            if not answer:
                answer = "I’m sorry, I could not generate a response right now."

            finalize_turn(
                session_id,
                user_message,
                answer,
                language_hint=language_hint,
                file_context=file_context,
                image_context=image_context,
                image_ocr=image_ocr,
                attachments=attachments,
                endpoint=endpoint,
                record_history=record_history,
                replace_last_assistant=replace_last_assistant,
            )

            payload = make_response_payload(
                session_id=session_id,
                user_message=user_message,
                answer=answer,
                language_hint=language_hint,
                attachments=attachments or [],
                file_context=file_context,
                image_context=image_context,
                image_ocr=image_ocr,
                endpoint=endpoint,
            )
            yield sse_data({"type": "done", **payload})

        except Exception as exc:
            yield sse_data(
                {
                    "type": "error",
                    "error": clean_text(str(exc)) or "Request failed.",
                }
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# -------------------------------------------------------------------
# Route wrappers
# -------------------------------------------------------------------
def handle_generation_request(route_name: str, multi_image: bool = False):
    sid = get_session_id()
    payload = request_payload()
    stream_mode = payload_bool(payload, "stream")

    user_message = normalize_message(
        get_payload_value(payload, "message", "msg", "caption", "text", default="")
    )

    file_storage = request.files.get("file")
    image_files = get_all_image_files(request) if multi_image else ([] if get_first_image_file(request) is None else [get_first_image_file(request)])

    try:
        file_context = ""
        image_context = ""
        image_ocr = ""
        attachments: List[Dict[str, Any]] = []

        if file_storage and file_storage.filename:
            file_context, file_attachment = prepare_file_context(file_storage, user_message)
            attachments.append(file_attachment)

        if image_files:
            image_context, image_ocr, image_attachments = prepare_image_context(image_files, user_message)
            attachments.extend(image_attachments)

        if not user_message and not attachments:
            return jsonify({"error": "Please send a message or attach a file/image."}), 400

        if stream_mode:
            return stream_answer(
                session_id=sid,
                user_message=user_message,
                file_context=file_context,
                image_context=image_context,
                image_ocr=image_ocr,
                attachments=attachments,
                endpoint=route_name,
                record_history=True,
                replace_last_assistant=False,
                exclude_last_assistant=False,
            )

        result = compose_answer(
            sid,
            user_message,
            file_context=file_context,
            image_context=image_context,
            image_ocr=image_ocr,
            attachments=attachments,
            endpoint=route_name,
            record_history=True,
        )
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": clean_text(str(exc)) or "Request failed."}), 500


# -------------------------------------------------------------------
# Views / APIs
# -------------------------------------------------------------------
@app.get("/")
def index():
    get_session_id()
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/clear")
def clear_history():
    sid = get_session_id()
    memory_store.clear(sid)
    memory_store.clear_turn_cache(sid)
    return jsonify({"ok": True})


@app.post("/transcribe")
def transcribe_audio():
    get_session_id()

    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file uploaded."}), 400

    suffix = Path(audio_file.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTS:
        return jsonify({"error": f"Unsupported audio type: {suffix}"}), 400

    temp_path = ""
    try:
        temp_path = save_upload(audio_file, AUDIO_DIR)
        language_hint = request.form.get("language") or None
        transcript = groq_service.transcribe_audio_file(temp_path, language=language_hint)
        transcript = clean_text(transcript)
        language = detect_language(transcript)
        return jsonify({"ok": True, "transcript": transcript, "language": language})
    except Exception as exc:
        return jsonify({"error": clean_text(str(exc)) or "Transcription failed."}), 500


@app.post("/get")
def get_text_reply():
    return handle_generation_request(route_name="get", multi_image=False)


@app.post("/chat")
def chat_combined():
    return handle_generation_request(route_name="chat", multi_image=True)


@app.post("/chat_stream")
def chat_stream():
    return handle_generation_request(route_name="chat", multi_image=True)


@app.post("/regenerate")
def regenerate_response():
    sid = get_session_id()
    payload = request_payload()
    stream_mode = payload_bool(payload, "stream")

    cached = memory_store.get_turn_cache(sid)
    if not cached or not cached.get("user_message"):
        return jsonify({"error": "Nothing to regenerate."}), 400

    user_message = normalize_message(cached.get("user_message", ""))
    file_context = cached.get("file_context", "") or ""
    image_context = cached.get("image_context", "") or ""
    image_ocr = cached.get("image_ocr", "") or ""
    attachments = cached.get("attachments", []) or []
    endpoint = cached.get("endpoint") or "chat"

    if stream_mode:
        return stream_answer(
            session_id=sid,
            user_message=user_message,
            file_context=file_context,
            image_context=image_context,
            image_ocr=image_ocr,
            attachments=attachments,
            endpoint=endpoint,
            record_history=False,
            replace_last_assistant=True,
            exclude_last_assistant=True,
        )

    result = compose_answer(
        sid,
        user_message,
        file_context=file_context,
        image_context=image_context,
        image_ocr=image_ocr,
        attachments=attachments,
        endpoint=endpoint,
        record_history=False,
        replace_last_assistant=True,
        exclude_last_assistant=True,
    )
    return jsonify(result)


@app.post("/followups")
def followups():
    sid = get_session_id()
    payload = request_payload()

    user_message = normalize_message(
        get_payload_value(payload, "user_message", "message", "msg", default="")
    )
    assistant_answer = normalize_message(
        get_payload_value(payload, "assistant_answer", "answer", "reply", default="")
    )
    file_context = normalize_message(get_payload_value(payload, "file_context", default=""))
    image_context = normalize_message(get_payload_value(payload, "image_context", default=""))
    image_ocr = normalize_message(get_payload_value(payload, "image_ocr", default=""))
    language_hint = get_payload_value(payload, "language", default="")

    if not user_message:
        cached = memory_store.get_turn_cache(sid) or {}
        user_message = normalize_message(cached.get("user_message", ""))
        assistant_answer = assistant_answer or normalize_message(cached.get("assistant_answer", ""))
        file_context = file_context or normalize_message(cached.get("file_context", ""))
        image_context = image_context or normalize_message(cached.get("image_context", ""))
        image_ocr = image_ocr or normalize_message(cached.get("image_ocr", ""))

    if not user_message:
        return jsonify({"error": "No context available for follow-up questions."}), 400

    if not language_hint:
        language_hint = detect_language(" ".join([user_message, assistant_answer, file_context, image_context, image_ocr]))

    summary, _, _ = build_generation_context(
        sid,
        user_message,
        file_context=file_context,
        image_context=image_context,
        image_ocr=image_ocr,
    )

    try:
        questions = groq_service.generate_followups(
            user_message=user_message,
            assistant_answer=assistant_answer,
            language=language_hint,
            memory_summary=summary,
            file_context=file_context,
            image_context=image_context,
            image_ocr=image_ocr,
        )
        return jsonify({"ok": True, "questions": questions or []})
    except Exception as exc:
        return jsonify({"error": clean_text(str(exc)) or "Follow-up generation failed."}), 500


@app.post("/analyze-images")
def analyze_images():
    sid = get_session_id()

    caption = normalize_message(
        get_payload_value(
            request_payload(),
            "caption",
            "message",
            "msg",
            "text",
            default="",
        )
    )

    image_files = get_all_image_files(request)
    if not image_files:
        return jsonify({"error": "No images uploaded."}), 400

    try:
        image_context, image_ocr, attachments = prepare_image_context(image_files, caption)
        result = compose_answer(
            sid,
            caption or "Please analyze the uploaded images and explain anything visible or medically relevant.",
            image_context=image_context,
            image_ocr=image_ocr,
            attachments=attachments,
            endpoint="chat",
            record_history=True,
        )
        result.update(
            {
                "caption_used": bool(caption),
                "images": attachments,
                "attachments": attachments,
                "image_used": True,
            }
        )
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": clean_text(str(exc)) or "Image analysis failed."}), 500


# -------------------------------------------------------------------
# Errors
# -------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Uploaded file is too large."}), 413


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    cleanup_old_uploads()
    start_cleanup_thread()
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "true").lower() == "true",
    )