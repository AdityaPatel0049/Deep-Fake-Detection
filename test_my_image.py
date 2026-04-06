"""
Simple script to test your own images with the fraud detection model.
Usage: python test_my_image.py "path/to/your/image.jpg"
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.model_loader import ModelLoader
from configs.config import MODEL_DIR

def test_image(image_path):
    """Test a single image and show results"""

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return

    # Check if it's a valid image file
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in valid_extensions:
        print(f" Error: Invalid file type. Supported: {valid_extensions}")
        return

    try:
        # Initialize model loader
        print("Loading models...")
        model_loader = ModelLoader(model_dir=MODEL_DIR)

        # Run ensemble prediction
        print(f" Analyzing: {Path(image_path).name}")
        predictions = model_loader.ensemble_predict(image_path)

        # Get results
        ensemble_class = predictions["ensemble_class"]
        ensemble_confidence = predictions["ensemble_confidence"]
        class_name = "REAL" if ensemble_class == 1 else "AI-GENERATED/FAKE"

        # Display results
        print("\n" + "="*60)
        print(" FRAUD DETECTION RESULTS")
        print("="*60)
        print(f" File: {Path(image_path).name}")
        print(f" Prediction: {class_name}")
        print(f" Confidence: {ensemble_confidence:.1%}")

        # Show probabilities
        probs = predictions["ensemble_probabilities"]
        print(f"\n Probabilities:")
        print(".1f")
        print(".1f")

        # Show individual model votes
        votes = predictions["votes"]
        print(f"\nModel Votes:")
        print(f"   FAKE: {votes['fake']}")
        print(f"   REAL: {votes['real']}")

        # Risk assessment
        fake_prob = probs["fake"]
        if fake_prob > 0.7:
            risk = "HIGH RISK "
            recommendation = "REJECT"
        elif fake_prob > 0.4:
            risk = "MEDIUM RISK "
            recommendation = "REVIEW"
        else:
            risk = "LOW RISK "
            recommendation = "APPROVE"

        print(f"\n Risk Level: {risk}")
        print(f"Recommendation: {recommendation}")

        print("="*60)

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Make sure your models are trained and available in the 'models/' directory")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_my_image.py \"path/to/your/image.jpg\"")
        print("\nExample:")
        print("  python test_my_image.py \"C:\\Users\\yourname\\Pictures\\photo.jpg\"")
        print("  python test_my_image.py \"data/test/real/0000 (1).jpg\"")
        sys.exit(1)

    image_path = sys.argv[1]
    test_image(image_path)