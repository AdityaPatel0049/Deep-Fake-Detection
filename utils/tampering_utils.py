import os
import cv2
import numpy as np

def analyze_tampering(image_path):
    """
    Performs Error Level Analysis (ELA) to find anomalies in JPEG compression,
    which often indicates splicing or copy-move forgery.
    Returns a dictionary containing tampering flags and a suspicion score (0.0 to 1.0).
    """
    flags = []
    suspicion_score = 0.0
    
    # We can only perform ELA reliably on JPEGs, but we can do a naive check if it opens.
    if not str(image_path).lower().endswith(('.jpg', '.jpeg')):
        flags.append("Image is not a JPEG. Format conversion may indicate source obfuscation.")
        suspicion_score += 0.3
        
    try:
        original = cv2.imread(image_path)
        if original is None:
            return {"score": 1.0, "flags": ["Invalid image file."]}

        # Save an intentionally highly-compressed version to compare
        temp_filename = "temp_ela_check.jpg"
        cv2.imwrite(temp_filename, original, [cv2.IMWRITE_JPEG_QUALITY, 90])
        compressed = cv2.imread(temp_filename)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        # Calculate pixel-wise absolute difference
        diff = cv2.absdiff(original, compressed)
        
        # Convert to grayscale to evaluate structural differences
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Calculate maximum pixel error
        # In a generic untouched photo, the error level should be relatively uniform.
        # High variance suggests areas were inserted from different sources.
        variance = np.var(gray_diff)
        max_diff = np.max(gray_diff)
        
        # Thresholds derived empirically for anomalies
        if max_diff > 45:
            flags.append(f"High compression delta detected (Max Diff: {max_diff}). Possible splice/edit.")
            suspicion_score += 0.4
            
        if variance > 30:
            flags.append(f"High compression variance detected across regions (Var: {variance:.2f}).")
            suspicion_score += 0.4
            
        if max_diff <= 45 and variance <= 30:
            flags.append("ELA indicates uniform compression (Normal text/structural integrity).")

    except Exception as e:
        flags.append(f"Error carrying out tampering analysis: {str(e)}")
        suspicion_score += 0.5
        
    return {
        "score": min(suspicion_score, 1.0),
        "flags": flags
    }
