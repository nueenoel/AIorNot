import os
from functools import lru_cache
from pathlib import Path

# Keep TensorFlow's CPU thread pools small for low-memory hosting.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

MODEL_ID = "kumaran-0188/image_forgery_detector"
MODEL_FILE = "forgery_model_fixed.keras"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE = 224

REAL_THRESHOLD = 0.65
FAKE_THRESHOLD = 0.35


@lru_cache(maxsize=1)
def _engine():
    import keras
    import tensorflow as tf
    from huggingface_hub import hf_hub_download

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE,
    )

    model = keras.saving.load_model(model_path, compile=False)
    return model, keras


def analyze_image(path: Path):
    from PIL import Image
    import numpy as np

    model, _keras = _engine()

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

    batch = np.expand_dims(array, axis=0)

    # The model card documents this output as the REAL probability.
    probability_real = float(model.predict(batch, verbose=0)[0][0])
    probability_real = max(0.0, min(1.0, probability_real))
    probability_fake = 1.0 - probability_real

    real_pct = round(probability_real * 100, 1)
    fake_pct = round(probability_fake * 100, 1)

    # Use a middle "uncertain" band instead of forcing a guess.
    if probability_real >= REAL_THRESHOLD:
        status = "LIKELY_REAL"
        label = "This image is real"
        confidence = round(probability_real * 100)
        message = "The image was classified as Real by the image forgery model."
    elif probability_real <= FAKE_THRESHOLD:
        status = "AI-GENERATED"
        label = "AI generated"
        confidence = round(probability_fake * 100)
        message = "The image was classified as Fake by the image forgery model."
    else:
        status = "UNCERTAIN"
        label = "Analysis inconclusive"
        confidence = round(max(probability_real, probability_fake) * 100)
        message = "The model did not have enough evidence to confidently classify this image as real or fake."

    return {
        "status": status,
        "label": label,
        "score": confidence,
        "fake_probability": fake_pct,
        "real_probability": real_pct,
        "frames_analyzed": 1,
        "model": MODEL_ID,
        "model_labels": {"0": "Fake", "1": "Real"},
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
