from flask import Flask, jsonify, render_template, request, send_from_directory
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from threading import Lock
import os
import time
import uuid
import tempfile
from typing import Iterable

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
UPLOAD = BASE / "uploads"
UPLOAD.mkdir(exist_ok=True)

# Per-file limits keep browser uploads practical on a 512 MB service.
IMAGE_MAX_BYTES = int(os.getenv("IMAGE_MAX_BYTES", 15 * 1024 * 1024))
VIDEO_MAX_BYTES = int(os.getenv("VIDEO_MAX_BYTES", 25 * 1024 * 1024))
AUDIO_MAX_BYTES = int(os.getenv("AUDIO_MAX_BYTES", 10 * 1024 * 1024))
UPLOAD_TOTAL_QUOTA_BYTES = int(os.getenv("UPLOAD_TOTAL_QUOTA_BYTES", 100 * 1024 * 1024))
UPLOAD_RETENTION_SECONDS = int(os.getenv("UPLOAD_RETENTION_SECONDS", 60 * 60))
MAX_REQUEST_BYTES = VIDEO_MAX_BYTES + (2 * 1024 * 1024)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}
AUDIO_EXTS = {"mp3", "wav"}
ALLOWED = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS
UPLOAD_LOCK = Lock()

AIORNOT_API_KEY = os.getenv("AIORNOT_API_KEY", "").strip()
AIORNOT_VOICE_ENDPOINT = "https://api.aiornot.com/v1/reports/voice"
AIORNOT_VIDEO_ENDPOINT = "https://api.aiornot.com/v2/video/sync"
REMOTE_TIMEOUT = (10, 120)
LOCAL_FRAME_LIMIT = 8


def extension(name: str) -> str:
    return name.rsplit(".", 1)[1].lower() if "." in name else ""


def allowed(name: str) -> bool:
    return extension(name) in ALLOWED


def file_kind(ext: str) -> str:
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return "unknown"


def max_bytes_for_kind(kind: str) -> int:
    return {
        "image": IMAGE_MAX_BYTES,
        "video": VIDEO_MAX_BYTES,
        "audio": AUDIO_MAX_BYTES,
    }[kind]


def cleanup_old_uploads() -> None:
    cutoff = time.time() - UPLOAD_RETENTION_SECONDS
    for item in UPLOAD.iterdir():
        try:
            if item.is_file() and item.stat().st_mtime < cutoff:
                item.unlink(missing_ok=True)
        except OSError:
            pass


def upload_directory_usage() -> int:
    total = 0
    for item in UPLOAD.iterdir():
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _stream_size(file_storage) -> int:
    stream = file_storage.stream
    try:
        position = stream.tell()
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(position)
        return int(size)
    except (AttributeError, OSError, ValueError):
        return -1


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise ValueError("That image could not be read. Please choose a valid JPG, PNG, or WEBP image.")


def save_upload(file_storage):
    cleanup_old_uploads()
    original = secure_filename(file_storage.filename or "")
    ext = extension(original)
    if not original or ext not in ALLOWED:
        raise ValueError("That file type isn't supported.")

    kind = file_kind(ext)
    per_file_limit = max_bytes_for_kind(kind)
    stream_size = _stream_size(file_storage)
    if stream_size == 0:
        raise ValueError("That file is empty.")
    if stream_size > per_file_limit:
        raise ValueError(
            f"That {kind} is larger than the {per_file_limit // (1024 * 1024)} MB {kind} limit."
        )

    with UPLOAD_LOCK:
        cleanup_old_uploads()
        current_usage = upload_directory_usage()
        if stream_size >= 0 and current_usage + stream_size > UPLOAD_TOTAL_QUOTA_BYTES:
            raise ValueError(
                "Temporary upload storage is full. Please try again after an earlier analysis finishes."
            )

        stored = f"{int(time.time())}_{uuid.uuid4().hex}.{ext}"
        path = UPLOAD / stored
        try:
            file_storage.save(path)
        except Exception:
            path.unlink(missing_ok=True)
            raise

        size = path.stat().st_size
        if size == 0:
            path.unlink(missing_ok=True)
            raise ValueError("That file is empty.")
        if size > per_file_limit:
            path.unlink(missing_ok=True)
            raise ValueError(
                f"That {kind} is larger than the {per_file_limit // (1024 * 1024)} MB {kind} limit."
            )
        if upload_directory_usage() > UPLOAD_TOTAL_QUOTA_BYTES:
            path.unlink(missing_ok=True)
            raise ValueError(
                "Temporary upload storage is full. Please try again after an earlier analysis finishes."
            )

    if kind == "image":
        try:
            validate_image(path)
        except ValueError:
            path.unlink(missing_ok=True)
            raise

    return original, stored, path, ext


def delete_upload(stored_name: str | None) -> None:
    if not stored_name:
        return
    safe_name = secure_filename(stored_name)
    if not safe_name:
        return
    try:
        (UPLOAD / safe_name).unlink(missing_ok=True)
    except OSError:
        pass


def analyze_image_path(path: Path):
    from detector import analyze_image
    return analyze_image(path)


def analyze_media(path: Path):
    from detector import analyze
    return analyze(path)


def classify_probability(fake_probability: float):
    if fake_probability >= 0.65:
        return "AI-GENERATED", "Likely AI-generated", round(fake_probability * 100)
    if fake_probability <= 0.35:
        return "LIKELY_REAL", "Likely real", round((1 - fake_probability) * 100)
    return "UNCERTAIN", "Analysis inconclusive", round(max(fake_probability, 1 - fake_probability) * 100)


def aggregate_frame_probabilities(probabilities: Iterable[float]):
    import statistics

    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("No frame results were supplied.")
    values.sort()
    median = statistics.median(values)
    if len(values) >= 5:
        trim = max(1, len(values) // 8)
        trimmed = values[trim:len(values) - trim]
        trimmed_mean = statistics.fmean(trimmed) if trimmed else median
    else:
        trimmed_mean = statistics.fmean(values)

    return {
        "median": median,
        "trimmed_mean": trimmed_mean,
        "minimum": min(values),
        "maximum": max(values),
        "spread": max(values) - min(values),
        "count": len(values),
    }


def build_frame_result(frame_results):
    aggregate = aggregate_frame_probabilities(
        result["fake_probability"] / 100.0 for result in frame_results
    )
    fake_probability = aggregate["median"]
    status, label, confidence = classify_probability(fake_probability)
    fake_pct = round(fake_probability * 100, 1)
    real_pct = round((1 - fake_probability) * 100, 1)

    if status == "AI-GENERATED":
        message = "The sampled video frames contain visual signals consistent with AI-generated content."
    elif status == "LIKELY_REAL":
        message = "The sampled video frames did not show strong AI-generation signals."
    else:
        message = "The sampled video frames produced mixed evidence near the decision boundary."

    return {
        "status": status,
        "label": label,
        "score": confidence,
        "fake_probability": fake_pct,
        "real_probability": real_pct,
        "frames_analyzed": aggregate["count"],
        "median_frame_fake_probability": round(aggregate["median"] * 100, 1),
        "trimmed_mean_frame_fake_probability": round(aggregate["trimmed_mean"] * 100, 1),
        "frame_probability_min": round(aggregate["minimum"] * 100, 1),
        "frame_probability_max": round(aggregate["maximum"] * 100, 1),
        "frame_confidence_spread": round(aggregate["spread"] * 100, 1),
        "method": "browser-frame-scan",
        "audio_analyzed": False,
        "temporal_manipulation_analyzed": False,
        "message": message,
        "disclaimer": (
            "Video scan uses browser-extracted visual frames only. It does not assess audio or temporal manipulation. "
            "This is an automated model estimate, not a guarantee of authenticity."
        ),
    }


def _json_error(response):
    try:
        body = response.json()
    except ValueError:
        body = {}
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail:
        return detail[:300]
    return f"Provider returned HTTP {response.status_code}."


def remote_voice_analysis(path: Path):
    if not AIORNOT_API_KEY:
        raise RuntimeError("Audio detection requires the configured AIorNot provider.")

    import requests

    try:
        with path.open("rb") as audio_file:
            response = requests.post(
                AIORNOT_VOICE_ENDPOINT,
                headers={"Authorization": f"Bearer {AIORNOT_API_KEY}"},
                files={"file": (path.name, audio_file)},
                timeout=REMOTE_TIMEOUT,
            )
    except requests.RequestException as exc:
        raise RuntimeError(f"Audio provider request failed: {exc.__class__.__name__}.") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Audio provider request failed: {_json_error(response)}")

    data = response.json()
    report = data.get("report", {})
    confidence = float(report.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    # The documented voice response exposes a confidence value rather than a
    # verdict field. Treat high confidence as an AI-voice signal and avoid
    # claiming a human source from a low score alone.
    if confidence >= 0.65:
        status, label, score = "AI-GENERATED", "Likely AI voice", round(confidence * 100)
    elif confidence <= 0.35:
        status, label, score = "LIKELY_REAL", "No strong AI-voice signal", round((1 - confidence) * 100)
    else:
        status, label, score = "UNCERTAIN", "Analysis inconclusive", round(max(confidence, 1 - confidence) * 100)

    return {
        "status": status,
        "label": label,
        "score": score,
        "voice_confidence": round(confidence * 100, 1),
        "provider": "AI or Not",
        "provider_report_id": data.get("id"),
        "message": "AI or Not voice analysis was used for this result.",
        "disclaimer": "Voice detection is provided by an external AIorNot service and can make mistakes.",
    }


def _component_signal(component_name, component):
    if not isinstance(component, dict):
        return None
    detected = bool(component.get("is_detected"))
    confidence = float(component.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    return {
        "name": component_name,
        "detected": detected,
        "confidence": round(confidence * 100, 1),
    }


def remote_video_analysis(path: Path):
    if not AIORNOT_API_KEY:
        raise RuntimeError("Cloud video detection is not configured.")

    import requests

    params = [
        ("only", "ai_video"),
        ("only", "ai_voice"),
        ("only", "ai_music"),
        ("only", "deepfake_video"),
    ]

    try:
        with path.open("rb") as video_file:
            response = requests.post(
                AIORNOT_VIDEO_ENDPOINT,
                headers={"Authorization": f"Bearer {AIORNOT_API_KEY}"},
                params=params,
                files={"video": (path.name, video_file)},
                timeout=REMOTE_TIMEOUT,
            )
    except requests.RequestException as exc:
        raise RuntimeError(f"Video provider request failed: {exc.__class__.__name__}.") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Video provider request failed: {_json_error(response)}")

    data = response.json()
    report = data.get("report", {})
    components = [
        _component_signal("AI video", report.get("ai_video")),
        _component_signal("AI voice", report.get("ai_voice")),
        _component_signal("AI music", report.get("ai_music")),
        _component_signal("Deepfake video", report.get("deepfake_video")),
    ]
    signals = [signal for signal in components if signal]
    detected = [signal for signal in signals if signal["detected"]]

    if detected:
        strongest = max(detected, key=lambda signal: signal["confidence"])
        score = int(round(strongest["confidence"]))
        status = "AI-GENERATED" if score >= 65 else "UNCERTAIN"
        label = "Likely AI-generated" if status == "AI-GENERATED" else "Analysis inconclusive"
        message = f"AIorNot detected an AI signal in: {', '.join(signal['name'] for signal in detected)}."
    elif signals:
        score = int(round(min(signal["confidence"] for signal in signals)))
        status = "LIKELY_REAL" if score >= 65 else "UNCERTAIN"
        label = "Likely real" if status == "LIKELY_REAL" else "Analysis inconclusive"
        message = "AIorNot did not detect an AI signal in the analyzed media components."
    else:
        score = None
        status = "UNCERTAIN"
        label = "Analysis inconclusive"
        message = "AIorNot returned no recognized component results."

    meta = report.get("meta", {}) or {}
    return {
        "status": status,
        "label": label,
        "score": score,
        "provider": "AI or Not",
        "provider_report_id": data.get("id"),
        "signals": signals,
        "duration_tested": meta.get("duration"),
        "audio_analyzed": meta.get("audio") == "processed",
        "video_analyzed": meta.get("video") == "processed",
        "deepfake_video_available": any(signal["name"] == "Deepfake video" for signal in signals),
        "message": message,
        "disclaimer": (
            "Video analysis is provided by an external AIorNot service. Longer videos may be analyzed only in part; "
            "the provider response reports the duration tested."
        ),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def api_upload():
    if "media" not in request.files or not request.files["media"].filename:
        return jsonify({"error": "Choose a file first."}), 400

    try:
        original, stored, path, ext = save_upload(request.files["media"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError:
        return jsonify({"error": "The upload could not be saved. Please try again."}), 500

    kind = file_kind(ext)
    return jsonify({
        "ok": True,
        "filename": original,
        "stored_name": stored,
        "url": f"/uploads/{stored}",
        "size": path.stat().st_size,
        "extension": ext,
        "kind": kind,
        "capabilities": {
            "remote_media": bool(AIORNOT_API_KEY),
            "local_video_frames": kind == "video",
            "can_analyze_now": kind == "image" or kind == "video" or (kind == "audio" and bool(AIORNOT_API_KEY)),
        },
    })


@app.post("/api/analyze")
def api_analyze():
    stored = secure_filename(request.form.get("stored_name", ""))
    original = request.form.get("filename") or stored
    if not stored:
        return jsonify({"error": "No uploaded file supplied."}), 400

    path = UPLOAD / stored
    if not path.exists() or path.is_dir():
        return jsonify({"error": "The uploaded file could not be found. Please upload it again."}), 404

    ext = extension(path.name)
    kind = file_kind(ext)
    try:
        if kind == "image":
            result = analyze_image_path(path)
        elif kind == "video":
            if AIORNOT_API_KEY:
                result = remote_video_analysis(path)
            else:
                return jsonify({
                    "status": "FRAME_SCAN_REQUIRED",
                    "message": "No cloud video provider is configured. Extract video frames in the browser and send them to /api/analyze-frames.",
                }), 409
        elif kind == "audio":
            result = remote_voice_analysis(path)
        else:
            return jsonify({"error": "That media type is not supported."}), 400
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        return jsonify({"error": "The analysis service could not process this file."}), 503
    finally:
        # Completed or failed analyses do not keep the original upload around.
        delete_upload(stored)

    result["filename"] = original
    return jsonify(result)


@app.post("/api/analyze-frames")
def api_analyze_frames():
    files = request.files.getlist("frames")
    if not files:
        return jsonify({"error": "No video frames were supplied."}), 400
    if len(files) > LOCAL_FRAME_LIMIT:
        return jsonify({"error": f"Send at most {LOCAL_FRAME_LIMIT} video frames."}), 400

    stored = secure_filename(request.form.get("stored_name", ""))
    original = request.form.get("filename") or stored
    frame_results = []
    temp_paths = []

    try:
        for index, frame in enumerate(files):
            if not frame.filename:
                continue
            raw_name = secure_filename(frame.filename) or f"frame_{index}.jpg"
            suffix = Path(raw_name).suffix.lower()
            if suffix not in {".jpg", ".jpeg"}:
                suffix = ".jpg"
            with tempfile.NamedTemporaryFile(prefix="aiornot_frame_", suffix=suffix, delete=False) as temp:
                temp_path = Path(temp.name)
            frame.save(temp_path)
            temp_paths.append(temp_path)
            validate_image(temp_path)
            frame_results.append(analyze_image_path(temp_path))

        if not frame_results:
            return jsonify({"error": "No valid frames were supplied."}), 400

        result = build_frame_result(frame_results)
        result["filename"] = original
        return jsonify(result)
    except (ValueError, UnidentifiedImageError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "The local video frame scan could not be completed."}), 503
    finally:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        delete_upload(stored)


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "service": "AIorNot",
        "image_analysis": "available",
        "capabilities": {
            "image": True,
            "video_local_frames": True,
            "video_remote": bool(AIORNOT_API_KEY),
            "audio_remote": bool(AIORNOT_API_KEY),
        },
        "limits_mb": {
            "image": round(IMAGE_MAX_BYTES / (1024 * 1024), 1),
            "video": round(VIDEO_MAX_BYTES / (1024 * 1024), 1),
            "audio": round(AUDIO_MAX_BYTES / (1024 * 1024), 1),
            "total": round(UPLOAD_TOTAL_QUOTA_BYTES / (1024 * 1024), 1),
        },
    })


@app.get("/uploads/<path:name>")
def uploaded(name):
    safe_name = secure_filename(Path(name).name)
    if not safe_name:
        return jsonify({"error": "File not found."}), 404
    return send_from_directory(UPLOAD, safe_name)


@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "error": (
            "That request is too large. Image/video/audio limits are "
            f"{IMAGE_MAX_BYTES // (1024 * 1024)} MB / "
            f"{VIDEO_MAX_BYTES // (1024 * 1024)} MB / "
            f"{AUDIO_MAX_BYTES // (1024 * 1024)} MB."
        )
    }), 413


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Something went wrong while processing the request."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 7860)), debug=os.getenv("DEBUG", "0") == "1")
