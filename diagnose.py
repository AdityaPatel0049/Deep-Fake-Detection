"""
Diagnostic script to check if training data is correct and if class labels are swapped.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.model_loader import ModelLoader
from configs.config import MODEL_DIR, IMAGE_SIZE
from torchvision import transforms
from PIL import Image
import torch

# Initialize model loader
model_loader = ModelLoader(model_dir=MODEL_DIR)

# Load ResNet model
print("Loading ResNet model...")
model = model_loader.load_model("resnet")

# Transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("\n" + "="*60)
print("DIAGNOSTIC TEST: Are Class Labels Swapped?")
print("="*60)

# Test on FAKE images (should predict as FAKE = class 0)
print("\n[TEST 1] Predicting on FAKE training images...")
print("Expected: All should predict as FAKE (class 0 / probability < 0.5)")
print("-" * 60)

fake_dir = "data/train/fake"
fake_images = list(Path(fake_dir).glob("*.jpg"))[:5]  # First 5 fake images

fake_predictions = []
for img_path in fake_images:
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            fake_prob = probs[0][0].item()  # Class 0 = Fake
            real_prob = probs[0][1].item()  # Class 1 = Real
            predicted_class = 0 if fake_prob > real_prob else 1
            predicted_label = "FAKE" if predicted_class == 0 else "REAL"
            
        fake_predictions.append(predicted_class)
        print(f"  {img_path.name}: Predicted={predicted_label} (Fake:{fake_prob:.4f}, Real:{real_prob:.4f})")
    except Exception as e:
        print(f"  Error processing {img_path.name}: {e}")

# Test on REAL images (should predict as REAL = class 1)
print("\n[TEST 2] Predicting on REAL training images...")
print("Expected: All should predict as REAL (class 1 / probability > 0.5)")
print("-" * 60)

real_dir = "data/train/real"
real_images = list(Path(real_dir).glob("*.jpg"))[:5]  # First 5 real images

real_predictions = []
for img_path in real_images:
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            fake_prob = probs[0][0].item()  # Class 0 = Fake
            real_prob = probs[0][1].item()  # Class 1 = Real
            predicted_class = 0 if fake_prob > real_prob else 1
            predicted_label = "FAKE" if predicted_class == 0 else "REAL"
            
        real_predictions.append(predicted_class)
        print(f"  {img_path.name}: Predicted={predicted_label} (Fake:{fake_prob:.4f}, Real:{real_prob:.4f})")
    except Exception as e:
        print(f"  Error processing {img_path.name}: {e}")

# Diagnosis
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

fake_correct = sum(1 for p in fake_predictions if p == 0)
real_correct = sum(1 for p in real_predictions if p == 1)

print(f"\nFake images: {fake_correct}/{len(fake_predictions)} predicted correctly as FAKE")
print(f"Real images: {real_correct}/{len(real_predictions)} predicted correctly as REAL")

if fake_correct == 0 and real_correct == 0:
    print("\n🔴 PROBLEM FOUND: Classes are SWAPPED!")
    print("   - FAKE images are being predicted as REAL")
    print("   - REAL images are being predicted as FAKE")
    print("\nSOLUTION: Swap class labels in prediction code")
    
elif fake_correct == len(fake_predictions) and real_correct == len(real_predictions):
    print("\n✅ Classes are CORRECT!")
    print("   Model is working properly on training data")
    print("   Issue might be with TEST data or dataset split")
    
else:
    print("\n⚠️  MIXED RESULTS: Uncertain classification")
    print("   This could indicate overfitting or data quality issues")

print("\n" + "="*60)
