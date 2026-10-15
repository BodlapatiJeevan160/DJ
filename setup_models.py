"""
setup_models.py
===============
One-time model setup for JARVIS V1.

OpenWakeWord (v0.6.0+) does NOT bundle wake-word model files with the pip
package.  The models are hosted as GitHub Release assets and must be
downloaded separately using openwakeword's own official API before the
voice engine can start.

Run this script once after `pip install -r requirements.txt`:

    python setup_models.py

The download is fully idempotent: files that are already present in the
package's resources/models/ directory are silently skipped.
No hard-coded user paths are used.
All models are stored locally inside the installed openwakeword package --
no cloud service is contacted at runtime.
"""

import sys


def main() -> None:
    try:
        import openwakeword
    except ImportError:
        print("[ERROR] openwakeword is not installed.  Run:")
        print("          pip install -r requirements.txt")
        sys.exit(1)

    version = getattr(openwakeword, "__version__", "unknown")
    print(f"openwakeword version : {version}")

    from openwakeword.utils import download_models
    import pathlib
    import os

    target_dir = os.path.join(
        pathlib.Path(openwakeword.__file__).parent.resolve(),
        "resources", "models"
    )
    print(f"Model directory      : {target_dir}")
    print()

    # ------------------------------------------------------------------ #
    # Download the Hey-Jarvis wake-word model plus the shared feature
    # models (melspectrogram + speech embedding) and the VAD model.
    # download_models() handles all of these in a single call.
    # ------------------------------------------------------------------ #
    print("Downloading 'hey_jarvis_v0.1' model (ONNX + tflite) ...")
    print("(Files already present will be skipped automatically.)")
    download_models(model_names=["hey_jarvis_v0.1"])

    # Verify the ONNX file is now present
    onnx_path = os.path.join(target_dir, "hey_jarvis_v0.1.onnx")
    if os.path.exists(onnx_path):
        size_kb = os.path.getsize(onnx_path) // 1024
        print(f"\n[OK] hey_jarvis_v0.1.onnx is present ({size_kb} KB)")
    else:
        print(f"\n[ERROR] hey_jarvis_v0.1.onnx was NOT found at: {onnx_path}")
        print("        Check your internet connection and try again.")
        sys.exit(1)

    # Verify the feature models
    for fname in ("melspectrogram.onnx", "embedding_model.onnx"):
        fpath = os.path.join(target_dir, fname)
        if os.path.exists(fpath):
            print(f"[OK] {fname} is present")
        else:
            print(f"[WARN] {fname} not found -- voice engine may fail to initialize")

    print()
    print("Model setup complete.  You can now start JARVIS:")
    print("    python main.py")


if __name__ == "__main__":
    main()
