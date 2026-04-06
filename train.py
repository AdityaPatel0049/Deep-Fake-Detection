import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import copy

# Add parent directory to path to allow importing configs
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import (
    MODEL_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMAGE_SIZE, 
    AVAILABLE_MODELS, CLASS_NAMES
)

DEBUG_MODE = False  # Changed from True to False for full training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(BASE_DIR, "data", "train")
test_path = os.path.join(BASE_DIR, "data", "test")
model_save_path = os.path.join(BASE_DIR, MODEL_DIR)

# Ensure directories exist
if not os.path.exists(train_path) or not os.path.exists(test_path):
    print(f"Error: Data directories not found. Please ensure {train_path} and {test_path} exist.")
    sys.exit(1)

os.makedirs(model_save_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_path, transform=train_transform)
test_data = datasets.ImageFolder(test_path, transform=val_transform)

if DEBUG_MODE:
    train_data = Subset(train_data, range(min(20000, len(train_data))))
    test_data = Subset(test_data, range(min(5000, len(test_data))))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

print("Classes:", CLASS_NAMES)
print("Training images:", len(train_data))
print("Testing images:", len(test_data))

def get_model(name):
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
    return model.to(device)

def train_and_evaluate(model, model_name, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Added weight decay for regularization
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Step the scheduler
        scheduler.step()
        
        # Evaluate at current epoch
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                preds.extend(predicted.cpu().numpy())
                targets.extend(labels.numpy())

        acc = accuracy_score(targets, preds)
        print(f"Epoch {epoch+1} Val Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Best Validation Accuracy for {model_name}: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, best_acc

results = {}

for model_name in AVAILABLE_MODELS:
    print(f"\nTraining {model_name.upper()}...")

    model = get_model(model_name)
    model, best_acc = train_and_evaluate(model, model_name)

    results[model_name] = best_acc

    save_path = os.path.join(model_save_path, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")

print("\nFINAL RESULTS:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")