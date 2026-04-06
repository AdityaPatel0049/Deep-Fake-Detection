import os
from PIL import Image
from PIL.ExifTags import TAGS

def analyze_metadata(image_path):
    """
    Extracts and analyzes EXIF metadata to determine the likelihood of forgery.
    Returns a dictionary of flags and a suspicion score (0.0 to 1.0).
    """
    flags = []
    suspicion_score = 0.0

    try:
        image = Image.open(image_path)
        
        # getexif() might return None on formats like PNG, or empty Dict on striped JPGs
        exif_data = image.getexif()

        if not exif_data:
            flags.append("Missing EXIF metadata completely (common in screenshotted or AI generated images).")
            suspicion_score += 0.8
            return {"score": min(suspicion_score, 1.0), "flags": flags}

        # Parse EXIF tags
        metadata = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            metadata[tag] = value

        # Check for suspicious software tags
        suspicious_software = ['photoshop', 'gimp', 'midjourney', 'dall-e', 'stable diffusion', 'canva']
        software = str(metadata.get('Software', '')).lower()
        if software:
            for suspect in suspicious_software:
                if suspect in software:
                    flags.append(f"Image edited or generated using: {software.title()}")
                    suspicion_score += 0.9
                    break
                    
        # Check if basic camera data is missing (often stripped during manipulation)
        if 'Make' not in metadata and 'Model' not in metadata:
            flags.append("Missing Camera Make/Model EXIF tags.")
            suspicion_score += 0.3

        if 'DateTimeOriginal' not in metadata:
            flags.append("Missing original capture timestamp.")
            suspicion_score += 0.2
            
        if len(flags) == 0:
            flags.append("EXIF data appears fundamentally normal.")

    except Exception as e:
        flags.append(f"Error reading image metadata: {str(e)}")
        suspicion_score += 0.5  # Suspicious if the file is malformed

    return {
        "score": min(suspicion_score, 1.0),
        "flags": flags
    }
