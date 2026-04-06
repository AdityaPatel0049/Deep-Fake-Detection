from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

MODEL_NAME = "umm-maybe/AI-image-detector"

# Quick pipeline test
pipe = pipeline("image-classification", model=MODEL_NAME)
result = pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")
print("Pipeline result:", result)

# Proper programmatic inference
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

image = Image.open("./data/test/real/0000 (10).jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

label_map = model.config.id2label
prediction_index = int(probs.argmax())
print("Label map:", label_map)
print("Probabilities:", {label_map[i]: float(probs[i]) for i in range(len(probs))})
print("Prediction:", label_map[prediction_index], "score:", float(probs[prediction_index]))
