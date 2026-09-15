import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

MODEL_REPO = "onnx-community/ai-image-detect-distilled-ONNX"
MODEL_FILE = "onnx/model_int8.onnx"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = 224

# The model's documented labels are:
# 0 = fake
# 1 = real
FAKE_INDEX = 0
REAL_INDEX = 1

# Do not force a binary answer near the middle.
FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.65


@lru_cache(maxsize=1)
def _engine():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session


def _preprocess(path: Path):
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

    # Model preprocessing: rescale to [0,1], then normalize with mean/std 0.5.
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32)


def analyze_image(path: Path):
    import numpy as np

    session = _engine()
    input_name = session.get_inputs()[0].name
    input_tensor = _preprocess(path)

    outputs = session.run(None, {input_name: input_tensor})
    logits = np.asarray(outputs[0])[0]

    # Stable softmax for the two-class logits.
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)

    fake_probability = float(probs[FAKE_INDEX])
    real_probability = float(probs[REAL_INDEX])

    fake_pct = round(fake_probability * 100, 1)
    real_pct = round(real_probability * 100, 1)

    if fake_probability >= FAKE_THRESHOLD:
        status = "AI-GENERATED"
        label = "AI generated"
        confidence = round(fake_probability * 100)
        message = "The image was classified as Fake by the image detection model."
    elif real_probability >= REAL_THRESHOLD:
        status = "LIKELY_REAL"
        label = "This image is real"
        confidence = round(real_probability * 100)
        message = "The image was classified as Real by the image detection model."
    else:
        status = "UNCERTAIN"
        label = "Analysis inconclusive"
        confidence = round(max(fake_probability, real_probability) * 100)
        message = (
            "The model did not have enough evidence to confidently classify "
            "this image as real or fake."
        )

    return {
        "status": status,
        "label": label,
        "score": confidence,
        "fake_probability": fake_pct,
        "real_probability": real_pct,
        "frames_analyzed": 1,
        "model": MODEL_REPO,
        "model_labels": {"0": "fake", "1": "real"},
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
