from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import os
import time
import uuid

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
UPLOAD = BASE / "uploads"
UPLOAD.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
UPLOAD_RETENTION_SECONDS = int(os.getenv("UPLOAD_RETENTION_SECONDS", 24 * 60 * 60))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}
AUDIO_EXTS = {"mp3", "wav"}
ALLOWED = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS


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


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise ValueError("That image could not be read. Please choose a valid JPG, PNG, or WEBP image.")


def cleanup_old_uploads():
    cutoff = time.time() - UPLOAD_RETENTION_SECONDS
    for item in UPLOAD.iterdir():
        try:
            if item.is_file() and item.stat().st_mtime < cutoff:
                item.unlink(missing_ok=True)
        except OSError:
            pass


def save_upload(file_storage):
    cleanup_old_uploads()
    original = secure_filename(file_storage.filename or "")
    ext = extension(original)
    if not original or ext not in ALLOWED:
        raise ValueError("That file type isn't supported.")

    stored = f"{int(time.time())}_{uuid.uuid4().hex}.{ext}"
    path = UPLOAD / stored
    file_storage.save(path)

    size = path.stat().st_size
    if size == 0:
        path.unlink(missing_ok=True)
        raise ValueError("That file is empty.")
    if size > MAX_UPLOAD_BYTES:
        path.unlink(missing_ok=True)
        raise ValueError(f"That file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    if ext in IMAGE_EXTS:
        try:
            validate_image(path)
        except ValueError:
            path.unlink(missing_ok=True)
            raise

    return original, stored, path, ext


def analyze_media(path: Path):
    from detector import analyze
    return analyze(path)


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

    return jsonify({
        "ok": True,
        "filename": original,
        "stored_name": stored,
        "url": f"/uploads/{stored}",
        "size": path.stat().st_size,
        "extension": ext,
        "kind": file_kind(ext),
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

    try:
        result = analyze_media(path)
    except Exception:
        return jsonify({
            "error": "The analysis service is not ready or could not process this image. Please make sure the required ML dependencies are installed and try again."
        }), 503

    result["filename"] = original
    result["file_url"] = f"/uploads/{stored}"
    return jsonify(result)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "AIorNot", "image_analysis": "available"})


@app.get("/uploads/<path:name>")
def uploaded(name):
    safe_name = secure_filename(Path(name).name)
    if not safe_name:
        return jsonify({"error": "File not found."}), 404
    return send_from_directory(UPLOAD, safe_name)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"That file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."}), 413


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Something went wrong while processing the request."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 7860)), debug=os.getenv("DEBUG", "0") == "1")
