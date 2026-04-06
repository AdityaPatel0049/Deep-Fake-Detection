import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path

# Add parent directory to path to allow importing configs and utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.video_utils import extract_frames_from_video
from configs.config import MAX_VIDEO_FRAMES_TO_EXTRACT, FRAME_INTERVAL

# Hugging Face imports
from transformers import AutoImageProcessor, AutoModelForImageClassification

class ModelLoader:
    def __init__(self, model_dir="models", use_huggingface=True):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.use_huggingface = use_huggingface
        
        # Standard torchvision transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Hugging Face model setup
        if self.use_huggingface:
            self.hf_model_name = "umm-maybe/AI-image-detector"
            self.hf_processor = AutoImageProcessor.from_pretrained(self.hf_model_name)
            self.hf_model = AutoModelForImageClassification.from_pretrained(self.hf_model_name).to(self.device)
            self.hf_model.eval()
        
    def load_model(self, name):
        """Load a trained model from disk"""
        if name in self.models:
            return self.models[name]
        
        model_path = os.path.join(self.model_dir, f"{name}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = self._get_base_model(name)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        self.models[name] = model
        return model
    
    def _get_base_model(self, name):
        """Get base model architecture"""
        if name == "resnet":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif name == "mobilenet":
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        elif name == "efficientnet":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        else:
            raise ValueError(f"Unknown model: {name}")
        
        return model.to(self.device)
    
    def predict_hf(self, image_input):
        """Predict using Hugging Face model"""
        # Load and preprocess image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported input type for prediction: {type(image_input)}")
        
        # Hugging Face preprocessing
        inputs = self.hf_processor(images=image, return_tensors="pt").to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.hf_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        
        # Map HF labels to our format: human=real(1), artificial=fake(0)
        label_map = self.hf_model.config.id2label
        predicted_label = label_map[probs.argmax()]
        predicted_class = 1 if predicted_label == "human" else 0  # human=real, artificial=fake
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(probs.max()),
            "probabilities": {
                "fake": float(probs[1] if predicted_label == "artificial" else probs[0]),  # artificial prob
                "real": float(probs[0] if predicted_label == "human" else probs[1])       # human prob
            },
            "hf_label": predicted_label,
            "hf_probabilities": {label_map[i]: float(probs[i]) for i in range(len(probs))}
        }
    
    def predict(self, image_input, model_name="resnet"):
        """Predict on an image (path, PIL Image, or numpy array)"""
        model = self.load_model(model_name)
        
        # Load and preprocess image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported input type for prediction: {type(image_input)}")
            
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0, predicted_class[0]].item()
        
        return {
            "predicted_class": predicted_class.item(),  # 0 = fake, 1 = real
            "confidence": confidence,
            "probabilities": {
                "fake": probabilities[0, 0].item(),
                "real": probabilities[0, 1].item()
            }
        }
    
    def ensemble_predict(self, image_input, model_names=None):
        """Ensemble prediction using local models or Hugging Face model"""
        if self.use_huggingface:
            # Use HF model directly and format as ensemble result
            hf_result = self.predict_hf(image_input)
            return {
                "ensemble_class": hf_result["predicted_class"],
                "ensemble_confidence": hf_result["confidence"],
                "ensemble_probabilities": {
                    "fake": hf_result["probabilities"]["fake"],
                    "real": hf_result["probabilities"]["real"]
                },
                "individual_predictions": [hf_result],  # Single prediction
                "votes": {
                    "fake": 1 if hf_result["predicted_class"] == 0 else 0,
                    "real": 1 if hf_result["predicted_class"] == 1 else 0
                }
            }
        else:
            # Use original torchvision models
            if model_names is None:
                model_names = ["resnet", "mobilenet", "efficientnet"]
            
            predictions = []
            for model_name in model_names:
                try:
                    pred = self.predict(image_input, model_name)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
            
            if not predictions:
                raise RuntimeError("No models available for prediction")
            
            # Average probabilities
            avg_fake = np.mean([p["probabilities"]["fake"] for p in predictions])
            avg_real = np.mean([p["probabilities"]["real"] for p in predictions])
            
            # Voting
            votes_fake = sum(1 for p in predictions if p["predicted_class"] == 0)
            votes_real = sum(1 for p in predictions if p["predicted_class"] == 1)
            
            ensemble_class = 1 if votes_real > votes_fake else 0
            ensemble_confidence = max(avg_fake, avg_real)
            
            return {
                "ensemble_class": ensemble_class,
                "ensemble_confidence": float(ensemble_confidence),
                "ensemble_probabilities": {
                    "fake": float(avg_fake),
                    "real": float(avg_real)
                },
                "individual_predictions": predictions,
                "votes": {
                    "fake": votes_fake,
                    "real": votes_real
                }
            }
    
    def predict_video(self, video_path, model_names=None):
        """Predict on an entire video by breaking it into frames and utilizing majority vote"""
        frames = extract_frames_from_video(
            video_path, 
            max_frames=MAX_VIDEO_FRAMES_TO_EXTRACT, 
            interval=FRAME_INTERVAL
        )
        
        if not frames:
            raise ValueError("No frames could be extracted from the video.")
            
        frame_predictions = []
        for i, frame in enumerate(frames):
            frame_pred = self.ensemble_predict(frame, model_names)
            frame_predictions.append(frame_pred)
            
        # Aggregate frame predictions
        avg_fake = np.mean([p["ensemble_probabilities"]["fake"] for p in frame_predictions])
        avg_real = np.mean([p["ensemble_probabilities"]["real"] for p in frame_predictions])
        
        total_fake_votes = sum(p["votes"]["fake"] if "votes" in p else (1 if p["ensemble_class"] == 0 else 0) for p in frame_predictions)
        total_real_votes = sum(p["votes"]["real"] if "votes" in p else (1 if p["ensemble_class"] == 1 else 0) for p in frame_predictions)
        
        overall_class = 1 if total_real_votes > total_fake_votes else 0
        overall_confidence = max(avg_fake, avg_real)
        
        return {
            "ensemble_class": overall_class,
            "ensemble_confidence": float(overall_confidence),
            "ensemble_probabilities": {
                "fake": float(avg_fake),
                "real": float(avg_real)
            },
            "frames_analyzed": len(frames),
            "votes": {
                "fake": int(total_fake_votes),
                "real": int(total_real_votes)
            }
        }
