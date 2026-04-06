"""
Test your own downloaded images with the Hugging Face AI detector.
Usage: python test_downloaded_image.py "C:\\path\\to\\your\\downloaded\\image.jpg"
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.model_loader import ModelLoader

def test_downloaded_image(image_path):
    """Test a downloaded image with the fraud detection system"""

    # Check if file exists
    if not os.path.exists(image_path):
        print(f" Error: Image file not found: {image_path}")
        print(" Make sure the path is correct and the file exists")
        return

    # Check if it's a valid image file
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in valid_extensions:
        print(f"❌ Error: Invalid file type. Supported: {valid_extensions}")
        print(f"   Your file: {file_ext}")
        return

    try:
        print("🔄 Loading Hugging Face AI detector model...")
        # Initialize model loader with Hugging Face
        model_loader = ModelLoader(use_huggingface=True)

        print(f"🔍 Analyzing: {Path(image_path).name}")
        predictions = model_loader.ensemble_predict(image_path)

        # Get results
        ensemble_class = predictions["ensemble_class"]
        ensemble_confidence = predictions["ensemble_confidence"]
        class_name = "REAL" if ensemble_class == 1 else "AI-GENERATED/FAKE"

        # Display results
        print("\n" + "="*60)
        print("🕵️  FRAUD DETECTION RESULTS")
        print("="*60)
        print(f"📁 File: {Path(image_path).name}")
        print(f"🎯 Prediction: {class_name}")
        print(f"📊 Confidence: {ensemble_confidence:.1%}")

        # Show probabilities
        probs = predictions["probabilities"]
        print(f"\n📈 Probabilities:")
        print(".1f")
        print(".1f")

        # Show HF details
        if "hf_label" in predictions:
            print(f"\n🤖 Hugging Face Label: {predictions['hf_label']}")

        # Risk assessment
        fake_prob = probs["fake"]
        if fake_prob > 0.7:
            risk = "HIGH RISK 🚨"
            recommendation = "REJECT"
        elif fake_prob > 0.4:
            risk = "MEDIUM RISK ⚠️"
            recommendation = "REVIEW"
        else:
            risk = "LOW RISK ✅"
            recommendation = "APPROVE"

        print(f"\n⚠️  Risk Level: {risk}")
        print(f"💡 Recommendation: {recommendation}")

        print("="*60)

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Make sure the model is properly loaded and the image is valid")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_downloaded_image.py \"C:\\\\path\\\\to\\\\your\\\\image.jpg\"")
        print("\nExample:")
        print("  python test_downloaded_image.py \"C:\\\\Users\\\\yourname\\\\Downloads\\\\photo.jpg\"")
        print("  python test_downloaded_image.py \"C:\\\\Pictures\\\\image.png\"")
        print("\nSupported formats: .jpg, .jpeg, .png, .gif, .bmp")
        sys.exit(1)

    image_path = sys.argv[1]
    test_downloaded_image(image_path)