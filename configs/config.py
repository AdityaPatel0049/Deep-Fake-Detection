# Configuration for Fraud Detection System

# Model Configuration
DEFAULT_MODEL = "resnet"
AVAILABLE_MODELS = ["resnet", "mobilenet", "efficientnet"]
MODEL_DIR = "models"

# Hugging Face integration
USE_HF_MODEL = True  # Changed to True since local models are not trained
HUGGINGFACE_MODEL_NAME = "umm-maybe/AI-image-detector"

# Data Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2  # 0: Fake/AI-generated, 1: Real
CLASS_NAMES = ["AI-Generated/Fake", "Real"]

# Training Configuration
EPOCHS = 3
LEARNING_RATE = 1e-4
OPTIMIZER = "adam"

# Fraud Scoring Configuration
FRAUD_SCORE_THRESHOLDS = {
    "LOW": (0, 40),      # Risk: LOW
    "MEDIUM": (40, 70),   # Risk: MEDIUM
    "HIGH": (70, 100)     # Risk: HIGH
}

RECOMMENDATIONS = {
    "LOW": "APPROVE",
    "MEDIUM": "REVIEW",
    "HIGH": "REJECT"
}

# File Upload Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
UPLOAD_FOLDER = "uploads"

# Video Processing Configuration
MAX_VIDEO_FRAMES_TO_EXTRACT = 10
FRAME_INTERVAL = 5  # Extract every 5th frame if possible

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "fraud_detection.log"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True
