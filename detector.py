from functools import lru_cache
from pathlib import Path

MODEL_ID = "king1oo1/deepfake-model"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@lru_cache(maxsize=1)
def _engine():
    from transformers import AutoProcessor, AutoModelForImageClassification
    import torch

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model, torch


def _class_indexes(model):
    labels = model.config.id2label or {}
    fake_idx = real_idx = None
    for i in range(model.config.num_labels):
        label = str(labels.get(i, "")).lower()
        if fake_idx is None and any(x in label for x in ("fake", "deepfake", "synthetic")):
            fake_idx = i
        if real_idx is None and any(x in label for x in ("real", "authentic", "genuine")):
            real_idx = i

    # This checkpoint is documented as class 0 = Real, class 1 = Fake.
    if real_idx is None:
        real_idx = 0
    if fake_idx is None:
        fake_idx = 1 if model.config.num_labels > 1 else 0
    return real_idx, fake_idx, labels


def analyze_image(path: Path):
    from PIL import Image

    processor, model, torch = _engine()
    with Image.open(path) as image:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    real_idx, fake_idx, labels = _class_indexes(model)
    real = float(probs[real_idx])
    fake = float(probs[fake_idx])
    total = real + fake
    if total:
        real /= total
        fake /= total

    fake_pct = round(fake * 100, 1)
    real_pct = round(real * 100, 1)
    confidence = round(max(fake, real) * 100)

    if fake >= real:
        status = "AI-GENERATED"
        label = "AI generated"
        message = "The image was classified closer to the model's Fake class."
    else:
        status = "LIKELY_REAL"
        label = "This image is real"
        message = "The image was classified closer to the model's Real class."

    return {
        "status": status,
        "label": label,
        "score": confidence,
        "fake_probability": fake_pct,
        "real_probability": real_pct,
        "frames_analyzed": 1,
        "model": MODEL_ID,
        "model_labels": {str(k): str(v) for k, v in labels.items()},
        "message": message,
        "disclaimer": "*Analysis is not 100% proficient and might make mistakes.",
    }


def analyze(path: Path):
    if path.suffix.lower() not in IMAGE_EXTS:
        return {
            "status": "COMING_SOON",
            "label": "Analysis coming soon",
            "score": None,
            "fake_probability": None,
            "real_probability": None,
            "signals": [],
            "message": "Image analysis is available now. Video and audio analysis are being added.",
            "disclaimer": "",
        }
    return analyze_image(path)
