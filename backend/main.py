import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.model_loader import ModelLoader
from configs.config import (
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, VIDEO_ALLOWED_EXTENSIONS, 
    MAX_FILE_SIZE, MODEL_DIR, FRAUD_SCORE_THRESHOLDS, 
    RECOMMENDATIONS, API_HOST, API_PORT, API_DEBUG, AVAILABLE_MODELS, DEFAULT_MODEL,
    USE_HF_MODEL, HUGGINGFACE_MODEL_NAME
)
from utils.metadata_utils import analyze_metadata
from utils.tampering_utils import analyze_tampering

# Create Flask app
app = Flask(__name__)
CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model loader
model_loader = ModelLoader(model_dir=MODEL_DIR, use_huggingface=USE_HF_MODEL)

def is_image_file(filename):
    """Check if file extension is an allowed image format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    """Check if file extension is an allowed video format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS

def calculate_composite_fraud_score(ai_predictions, metadata_res, tampering_res, is_video=False):
    """
    Calculate composite fraud risk score (0-100%).
    Weights: 60% AI Ensemble, 20% Metadata, 20% Tampering ELA.
    (If Video, relies mainly on AI since EXIF/JPEG ELA don't apply).
    """
    ai_fake_prob = ai_predictions.get("ensemble_probabilities", {}).get("fake", 0.5)
    
    if is_video:
        # Video relies entirely on the AI frame analysis right now
        return round(ai_fake_prob * 100, 2)
        
    ai_score = ai_fake_prob * 100.0
    meta_score = metadata_res.get("score", 0.0) * 100.0
    tamp_score = tampering_res.get("score", 0.0) * 100.0
    
    composite = (0.6 * ai_score) + (0.2 * meta_score) + (0.2 * tamp_score)
    return round(composite, 2)

def get_risk_mapping(fraud_score):
    """Maps score to Risk Level and Recommendation using config"""
    risk_level = "LOW"
    for level, (low, high) in FRAUD_SCORE_THRESHOLDS.items():
        if low <= fraud_score <= high:
            risk_level = level
            break
            
    recommendation = RECOMMENDATIONS.get(risk_level, "REVIEW")
    return risk_level, recommendation

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict if an image or video is real or AI-generated/manipulated.
    Expected: multipart/form-data with 'media' file
    """
    try:
        # Check if media is provided
        file_key = 'media' if 'media' in request.files else 'image'
        if file_key not in request.files:
            return jsonify({"error": "No media (or image) file provided"}), 400
        
        file = request.files[file_key]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        is_img = is_image_file(file.filename)
        is_vid = is_video_file(file.filename)
        
        if not is_img and not is_vid:
            return jsonify({"error": f"File type not allowed. Supported formats: Images {ALLOWED_EXTENSIONS}, Videos {VIDEO_ALLOWED_EXTENSIONS}"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        metadata_res = {"score": 0.0, "flags": []}
        tampering_res = {"score": 0.0, "flags": []}
        
        # Determine prediction route and secondary signals
        if is_vid:
            predictions = model_loader.predict_video(file_path)
            metadata_res["flags"].append("Metadata EXIF analysis bypassed for video container.")
            tampering_res["flags"].append("Error Level Analysis bypassed for video sequence.")
        else:
            if USE_HF_MODEL:
                predictions = model_loader.predict_hf(file_path)
            else:
                predictions = model_loader.ensemble_predict(file_path)
            metadata_res = analyze_metadata(file_path)
            tampering_res = analyze_tampering(file_path)
            
        # Calculate fraud score
        fraud_score = calculate_composite_fraud_score(predictions, metadata_res, tampering_res, is_vid)
        risk_level, recommendation = get_risk_mapping(fraud_score)
        
        response = {
            "status": "success",
            "media_type": "video" if is_vid else "image",
            "predictions": predictions,
            "metadata_analysis": metadata_res,
            "tampering_analysis": tampering_res,
            "fraud_score": fraud_score,
            "score_breakdown": {
                "ai_weight": 0.60 if not is_vid else 1.0,
                "metadata_weight": 0.20 if not is_vid else 0.0,
                "tampering_weight": 0.20 if not is_vid else 0.0
            },
            "risk_level": risk_level,
            "recommendation": recommendation,
            "class_names": {
                0: "AI-Generated/Fake",
                1: "Real"
            }
        }
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-single', methods=['POST'])
def predict_single():
    """
    Predict using a single model on an image
    Expected: multipart/form-data with 'image' file and 'model' parameter
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        model_name = request.form.get('model', DEFAULT_MODEL)
        
        if not is_image_file(file.filename):
            return jsonify({"error": "Only images are supported for single model prediction."}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get prediction
        if model_name.lower() in {"hf", "huggingface"}:
            prediction = model_loader.predict_hf(file_path)
        else:
            prediction = model_loader.predict(file_path, model_name)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            "status": "success",
            "model": model_name,
            "prediction": prediction
        }), 200
    
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models = AVAILABLE_MODELS.copy()
    if USE_HF_MODEL and HUGGINGFACE_MODEL_NAME not in models:
        models.append("huggingface")
    return jsonify({
        "available_models": models,
        "default_model": DEFAULT_MODEL,
        "huggingface_model": HUGGINGFACE_MODEL_NAME if USE_HF_MODEL else None
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=API_DEBUG, host=API_HOST, port=API_PORT)
