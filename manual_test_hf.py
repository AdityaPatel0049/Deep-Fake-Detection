"""
Manual Test Script for Hugging Face Model: umm-maybe/AI-image-detector
---------------------------------------------------------------------------
Tests the HF model directly (same logic as transformer.py / model_loader.py)
without needing the Flask backend or any locally trained models.

Run:
    python manual_test_hf.py
"""

import os
import sys
import time
from pathlib import Path

print("=" * 60)
print("  HF Model Manual Test: umm-maybe/AI-image-detector")
print("=" * 60)

# ── 1. Import check ──────────────────────────────────────────────
print("\n[1/5] Checking imports...")
try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from PIL import Image
    import numpy as np
    print(f"  ✓ torch       {torch.__version__}")
    import transformers
    print(f"  ✓ transformers {transformers.__version__}")
    print(f"  ✓ Pillow      OK")
    print(f"  ✓ numpy       OK")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("    Run: pip install torch transformers Pillow numpy")
    sys.exit(1)

# ── 2. Device info ───────────────────────────────────────────────
print("\n[2/5] Device / hardware info...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {device}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ── 3. Load model from Hugging Face ─────────────────────────────
MODEL_NAME = "umm-maybe/AI-image-detector"
print(f"\n[3/5] Loading model '{MODEL_NAME}' from Hugging Face...")
print("  (First run will download ~350 MB – cached afterwards)")
t0 = time.time()
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    elapsed = time.time() - t0
    print(f"  ✓ Model loaded in {elapsed:.1f}s")
    label_map = model.config.id2label
    print(f"  ✓ Label map: {label_map}")
except Exception as e:
    print(f"  ✗ Model load failed: {e}")
    sys.exit(1)

# ── 4. Inference helper ──────────────────────────────────────────
def predict_image(image_path: str) -> dict:
    """Run inference on a single image file."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_label = label_map[pred_idx]
    is_real = pred_label == "human"
    return {
        "path": image_path,
        "hf_label": pred_label,
        "is_real": is_real,
        "confidence": float(probs.max()),
        "probabilities": {label_map[i]: float(probs[i]) for i in range(len(probs))},
    }

def print_result(r: dict, expected: str = None):
    verdict = "REAL" if r["is_real"] else "FAKE/AI"
    icon = "✓" if (expected is None or
                   (expected == "real" and r["is_real"]) or
                   (expected == "fake" and not r["is_real"])) else "✗"
    print(f"  {icon}  [{verdict:8s}]  conf={r['confidence']:.2%}  "
          f"probs={{{', '.join(f'{k}: {v:.3f}' for k, v in r['probabilities'].items())}}}  "
          f"file={Path(r['path']).name}")

# ── 5. Run tests ─────────────────────────────────────────────────
print("\n[4/5] Running inference on local test images...")

FAKE_DIR = Path("data/test/fake")
REAL_DIR = Path("data/test/real")

fake_images = list(FAKE_DIR.glob("*.jpg"))[:5] + list(FAKE_DIR.glob("*.png"))[:5]
real_images = list(REAL_DIR.glob("*.jpg"))[:5] + list(REAL_DIR.glob("*.png"))[:5]

# Limit to 5 each
fake_images = fake_images[:5]
real_images = real_images[:5]

correct = 0
total = 0

if fake_images:
    print(f"\n  --- FAKE / AI-generated images (expected: FAKE) ---")
    for img_path in fake_images:
        try:
            r = predict_image(str(img_path))
            print_result(r, expected="fake")
            if not r["is_real"]:
                correct += 1
            total += 1
        except Exception as e:
            print(f"  ✗  ERROR on {img_path.name}: {e}")
else:
    print("  (!) No fake images found in data/test/fake/")

if real_images:
    print(f"\n  --- REAL images (expected: REAL) ---")
    for img_path in real_images:
        try:
            r = predict_image(str(img_path))
            print_result(r, expected="real")
            if r["is_real"]:
                correct += 1
            total += 1
        except Exception as e:
            print(f"  ✗  ERROR on {img_path.name}: {e}")
else:
    print("  (!) No real images found in data/test/real/")

# ── 6. Quick URL test (internet check) ──────────────────────────
ONLINE_TEST_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png"
print(f"\n[5/5] Quick pipeline test with an online image...")
try:
    from transformers import pipeline as hf_pipeline
    import urllib.request, io
    with urllib.request.urlopen(ONLINE_TEST_URL, timeout=10) as resp:
        img_bytes = resp.read()
    online_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = processor(images=online_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    pred_label = label_map[int(probs.argmax())]
    print(f"  ✓ Online image → '{pred_label}'  (probs: {{{', '.join(f'{label_map[i]}: {probs[i]:.3f}' for i in range(len(probs)))}}})")
    print(f"    (A real photo of parrots — expect 'human')")
except Exception as e:
    print(f"  (!) Online test skipped / failed: {e}")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
if total > 0:
    accuracy = correct / total * 100
    print(f"  Local images : {correct}/{total} correct  ({accuracy:.1f}% accuracy)")
    if accuracy >= 70:
        print("  ✓ Model appears to be working correctly!")
    else:
        print("  ⚠ Accuracy lower than expected — check label mapping.")
else:
    print("  No local images were tested.")
    print("  ✓ Model loaded and ran inference successfully (online test).")

print("=" * 60)
print("\nDone. You can also test a custom image by editing this line:")
print("  predict_image('path/to/your/image.jpg')")
