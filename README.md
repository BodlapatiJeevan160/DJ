<<<<<<< HEAD
# JARVIS V1

A fully local AI assistant with voice wake-word detection.

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download wake-word model files  *(required once)*

OpenWakeWord does **not** bundle model files with the pip package.
Run the setup script to download the Hey-Jarvis ONNX model from the
official GitHub Release assets into the local package directory:

```bash
python setup_models.py
```

This step is **idempotent** — files already present are skipped.
No cloud service is contacted at runtime after this step.

### 3. Run JARVIS

```bash
python main.py
```

## Wake-Word

JARVIS listens for **"Hey Jarvis"** using the `openwakeword` library.
All detection runs locally with no cloud API calls.

## Voice Input

Voice input uses:
- **openWakeWord** (`hey_jarvis_v0.1.onnx`) — local wake-word detection
- **faster-whisper** (tiny model, CPU/int8) — local transcription
- **PyAudio** — microphone capture

All processing is on-device. No audio is sent to any cloud service.

## Model Storage

After running `python setup_models.py`, models are stored inside the
installed `openwakeword` package at:

```
<venv>/Lib/site-packages/openwakeword/resources/models/
```

This location is resolved at runtime by the package itself — no
hard-coded user paths are used anywhere in the project code.
=======
# DJ
>>>>>>> origin/main
