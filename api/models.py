import json
from configs.config import AVAILABLE_MODELS, DEFAULT_MODEL, USE_HF_MODEL, HUGGINGFACE_MODEL_NAME

def handler(request):
    """Get available models and configuration"""
    try:
        response = {
            "available_models": AVAILABLE_MODELS + (["huggingface"] if USE_HF_MODEL else []),
            "default_model": DEFAULT_MODEL,
            "huggingface_model": HUGGINGFACE_MODEL_NAME if USE_HF_MODEL else None
        }
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps(response)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }