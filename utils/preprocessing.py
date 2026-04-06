import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from torchvision import transforms


def resize_image(image_path, size=(224, 224)):
    """Resize image to specified size"""
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    return img_resized


def normalize_image(image_array, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize image using ImageNet statistics"""
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array) / 255.0
    
    image_array = (image_array - mean) / std
    return image_array


def augment_image(image_path, augmentation_factor=5):
    """Apply data augmentation to an image"""
    img = Image.open(image_path).convert('RGB')
    augmented_images = []
    
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize((224, 224)),
    ])
    
    for _ in range(augmentation_factor):
        augmented = augmentation_transforms(img)
        augmented_images.append(augmented)
    
    return augmented_images


def check_image_quality(image_path):
    """Check image quality metrics"""
    img = cv2.imread(image_path)
    if img is None:
        return {"valid": False}
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance (sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Image size
    height, width = img.shape[:2]
    
    # Check for blur
    is_sharp = laplacian_var > 100
    is_large_enough = width >= 64 and height >= 64
    
    return {
        "valid": True,
        "sharpness": laplacian_var,
        "is_sharp": is_sharp,
        "dimensions": (width, height),
        "is_large_enough": is_large_enough,
        "quality_score": min(laplacian_var / 1000, 1.0)
    }


def get_image_histogram(image_path):
    """Get histogram features of image"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    colors = ('b', 'g', 'r')
    histograms = {}
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        histograms[color] = hist.flatten().tolist()
    
    return histograms


def detect_duplicates(image_path, reference_images_dir):
    """Use SIFT to detect if image is a duplicate"""
    sift = cv2.SIFT_create()
    
    # Query image
    query_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(query_img, None)
    
    matches_list = []
    
    for ref_img_path in os.listdir(reference_images_dir):
        ref_img = cv2.imread(os.path.join(reference_images_dir, ref_img_path), cv2.IMREAD_GRAYSCALE)
        kp2, des2 = sift.detectAndCompute(ref_img, None)
        
        if des2 is None or des1 is None:
            continue
        
        # Use BFMatcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        match_ratio = len(good_matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        matches_list.append({
            "image": ref_img_path,
            "matches": len(good_matches),
            "match_ratio": match_ratio
        })
    
    return sorted(matches_list, key=lambda x: x["match_ratio"], reverse=True)


def get_dominant_colors(image_path, k=3):
    """Extract dominant colors using K-means"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Reshape image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    return centers.tolist()


def is_valid_image(image_path):
    """Check if file is a valid image"""
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False
