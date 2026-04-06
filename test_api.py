import requests
import os

# Test the API endpoints
API_BASE = "http://127.0.0.1:5000"

print("Testing Fraud Detection API...")

# Test 1: Get available models
print("\n1. Testing /api/models endpoint...")
try:
    response = requests.get(f"{API_BASE}/api/models")
    if response.status_code == 200:
        print("✓ Models endpoint working")
        print("Available models:", response.json())
    else:
        print("✗ Models endpoint failed:", response.status_code)
except Exception as e:
    print("✗ Error:", e)

# Test 2: Try to test prediction with a dummy file
print("\n2. Testing prediction endpoint...")
test_image_path = "test_real.jpg"
if os.path.exists(test_image_path):
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{API_BASE}/api/predict", files=files)
            if response.status_code == 200:
                print("✓ Prediction endpoint working")
                result = response.json()
                print("Prediction result:", result)
            else:
                print("✗ Prediction endpoint failed:", response.status_code, response.text)
    except Exception as e:
        print("✗ Error:", e)
else:
    print("✗ No test image found")

print("\nAPI testing complete!")