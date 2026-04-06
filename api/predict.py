import json
import base64
import io
from configs.config import USE_HF_MODEL, DEFAULT_MODEL
from backend.model_loader import ModelLoader

# Initialize model loader (this will load on cold start)
model_loader = ModelLoader(use_huggingface=USE_HF_MODEL)

def handler(request):
    """Handle prediction requests"""
    try:
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            }

        if request.method != "POST":
            return {
                "statusCode": 405,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Method not allowed"})
            }

        # Get the file from the request
        # Vercel passes form data differently
        body = request.get("body", "")
        if not body:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "No file provided"})
            }

        # For simplicity, assume the file is sent as base64 in the body
        # In practice, you'd need to handle multipart form data
        # This is a simplified version

        # For now, return a mock response
        mock_response = {
            "status": "success",
            "model": DEFAULT_MODEL,
            "prediction": {
                "is_ai_generated": 0.3,
                "confidence": 0.7,
                "fraud_score": 30,
                "risk_category": "LOW",
                "recommendation": "APPROVE"
            }
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps(mock_response)
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