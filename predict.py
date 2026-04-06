import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.model_loader import ModelLoader
from configs.config import (
    FRAUD_SCORE_THRESHOLDS, RECOMMENDATIONS, MODEL_DIR, 
    ALLOWED_EXTENSIONS, VIDEO_ALLOWED_EXTENSIONS, AVAILABLE_MODELS, DEFAULT_MODEL,
    HUGGINGFACE_MODEL_NAME, USE_HF_MODEL
)
from utils.metadata_utils import analyze_metadata
from utils.tampering_utils import analyze_tampering

def calculate_composite_fraud_score(ai_predictions, metadata_res, tampering_res, is_video=False):
    """
    Calculate composite fraud risk score (0-100%).
    Weights: 60% AI Ensemble, 20% Metadata, 20% Tampering ELA.
    (If Video, relies mainly on AI since EXIF/JPEG ELA don't apply).
    """
    ai_fake_prob = ai_predictions.get("ensemble_probabilities", {}).get("fake", 0.5)
    
    if is_video:
        return round(ai_fake_prob * 100, 2)
        
    ai_score = ai_fake_prob * 100.0
    meta_score = metadata_res.get("score", 0.0) * 100.0
    tamp_score = tampering_res.get("score", 0.0) * 100.0
    
    composite = (0.6 * ai_score) + (0.2 * meta_score) + (0.2 * tamp_score)
    return round(composite, 2)

def get_risk_level(fraud_score):
    """Get risk level based on fraud score"""
    risk_level = "LOW"
    for level, (low, high) in FRAUD_SCORE_THRESHOLDS.items():
        if low <= fraud_score <= high:
            risk_level = level
            break
    return risk_level

def is_image_file(filename):
    """Check if file extension is an allowed image format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    """Check if file extension is an allowed video format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS

def main():
    parser = argparse.ArgumentParser(
        description="Predict if an image or video is real or AI-generated/manipulated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py path/to/image.jpg
  python predict.py path/to/video.mp4
  python predict.py path/to/image.jpg --model resnet
  python predict.py path/to/image.jpg --single-model
  python predict.py path/to/image.jpg --verbose
        """
    )
    
    parser.add_argument("media", help="Path to the image or video file")
    parser.add_argument(
        "--model", 
        default=DEFAULT_MODEL, 
        choices=AVAILABLE_MODELS,
        help=f"Model to use for single model prediction (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Use single model prediction instead of ensemble (Only applicable to images)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output including metadata and tampering flags."
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Use the Hugging Face image classifier model instead of local models."
    )
    
    args = parser.parse_args()
    
    # Validate file path
    if not os.path.exists(args.media):
        print(f"Error: media file not found: {args.media}")
        sys.exit(1)
        
    is_img = is_image_file(args.media)
    is_vid = is_video_file(args.media)
    
    if not is_img and not is_vid:
        print(f"Error: Unsupported file format. Please provide a valid image ({ALLOWED_EXTENSIONS}) or video ({VIDEO_ALLOWED_EXTENSIONS}).")
        sys.exit(1)
    
    try:
        # Initialize model loader
        model_loader = ModelLoader(model_dir=MODEL_DIR, use_huggingface=USE_HF_MODEL)
        
        if args.hf:
            if not is_img:
                print("Error: Hugging Face model only supports image files.")
                sys.exit(1)
            print(f"Running Hugging Face inference with {HUGGINGFACE_MODEL_NAME}...")
            prediction = model_loader.predict_hf(args.media)

            class_name = "Real" if prediction["predicted_class"] == 1 else "AI-Generated/Fake"
            print(f"\nPrediction: {class_name}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            print(f"HF Label: {prediction['predicted_label']}")
            if args.verbose:
                print(f"HF Probabilities: {prediction['hf_probabilities']}")
            sys.exit(0)
        
        if is_img and args.single_model:
            # Single model prediction (Image Only - ignores composite scoring)
            print(f"Running single model prediction with {args.model} on image...")
            prediction = model_loader.predict(args.media, args.model)
            
            class_name = "Real" if prediction["predicted_class"] == 1 else "AI-Generated/Fake"
            print(f"\nPrediction: {class_name}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            
            if args.verbose:
                print(f"Probabilities: {prediction['probabilities']}")
        
        else:
            # Ensemble prediction && Full Pipeline (Image or Video)
            metadata_res = {"score": 0.0, "flags": []}
            tampering_res = {"score": 0.0, "flags": []}
            
            if is_vid:
                if args.single_model:
                    print("Warning: --single-model passed with a video file. Video analysis uses ensemble. Proceeding with ensemble...")
                print("Running video ensemble prediction...")
                predictions = model_loader.predict_video(args.media)
                metadata_res["flags"].append("Bypassed analysis for Video.")
                tampering_res["flags"].append("Bypassed ELA for Video.")
            else:
                print("Running full image analysis pipeline (AI Ensemble + Metadata + Tampering ELA)...")
                predictions = model_loader.ensemble_predict(args.media)
                metadata_res = analyze_metadata(args.media)
                tampering_res = analyze_tampering(args.media)
                
            fraud_score = calculate_composite_fraud_score(predictions, metadata_res, tampering_res, is_vid)
            risk_level = get_risk_level(fraud_score)
            recommendation = RECOMMENDATIONS.get(risk_level, "REVIEW")
            
            # Output
            print("\n" + "=" * 50)
            print("FRAUD DETECTION RESULTS")
            print("=" * 50)
            
            ensemble_class = "Real" if predictions["ensemble_class"] == 1 else "AI-Generated/Fake"
            print(f"AI Ensemble Prediction: {ensemble_class}")
            print(f"AI Confidence: {predictions['ensemble_confidence']:.2%}")
            print(f"\nFinal Fraud Score: {fraud_score:.1f}%")
            print(f"Risk Level: {risk_level}")
            print(f"Recommendation: {recommendation}")
            
            if args.verbose:
                print("\n" + "-" * 50)
                print("DETAILED RESULTS")
                print("-" * 50)
                print(f"AI Model Breakdown:")
                print(f"  - Fake Prob: {predictions['ensemble_probabilities']['fake']:.4f}")
                print(f"  - Real Prob: {predictions['ensemble_probabilities']['real']:.4f}")
                
                if not is_vid:
                    # Metadata Breakdown
                    print(f"\nMetadata Analysis (Score: {metadata_res['score']*100:.1f}%):")
                    for flag in metadata_res['flags']:
                        print(f"  [!] {flag}")
                    
                    # Tampering Breakdown
                    print(f"\nTampering ELA Analysis (Score: {tampering_res['score']*100:.1f}%):")
                    for flag in tampering_res['flags']:
                        print(f"  [!] {flag}")
                        
                    print(f"\nIndividual Model Votes:")
                    print(f"  - Fake: {predictions['votes']['fake']}")
                    print(f"  - Real: {predictions['votes']['real']}")
                else:
                    print(f"\nFrames Analyzed: {predictions['frames_analyzed']}")
                    print(f"Frame Votes -> Fake: {predictions['votes']['fake']}, Real: {predictions['votes']['real']}")
        
        print("\n" + "=" * 50 + "\n")
    
    except FileNotFoundError as e:
        print(f"Error: Model file not found. Have you trained the models yet?")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
