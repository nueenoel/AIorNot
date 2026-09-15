# AIorNot

AIorNot analyzes uploaded images and estimates whether they are more likely to be AI-generated/manipulated or real. Video and audio uploads are accepted by the interface but their analysis is currently unavailable.

## Run locally

```bash
chmod +x run.sh
./run.sh
```

Then open `http://localhost:7860`.

The first image analysis downloads the pretrained Hugging Face model defined in `detector.py`, so the first request can take longer and requires internet access. The model is cached after the first load.

## Production

Set `PRODUCTION=1` to run with Gunicorn:

```bash
PRODUCTION=1 ./run.sh
```

The service exposes:

- `GET /` — web interface
- `POST /api/upload` — upload and validate a supported file
- `POST /api/analyze` — analyze an uploaded image
- `GET /api/health` — service health check

Uploads are stored temporarily and old files are cleaned automatically.

## Supported formats

Images: JPG, JPEG, PNG, WEBP

Video/audio: MP4, MOV, AVI, MKV, WEBM, MP3, WAV (upload supported; analysis to be added)

## Detection engine

AIorNot uses `king1oo1/deepfake-model`, a pretrained SigLIP2 image classifier whose model card describes two classes, `Real` and `Fake`, and reports 78.5% validation accuracy on its held-out set. The result shown by AIorNot is an automated estimate and is not a guarantee of authenticity.
