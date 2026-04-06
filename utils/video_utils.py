import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path


def extract_frames_from_video(video_path, output_dir=None, max_frames=30, interval=1):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames (optional)
        max_frames: Maximum number of frames to extract
        interval: Extract every nth frame
    
    Returns:
        List of frame arrays or paths if output_dir specified
    """
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        if frame_count % interval == 0 and extracted_count < max_frames:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                frame_path = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
            else:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            extracted_count += 1
        
        frame_count += 1
    
    video.release()
    return frames


def get_video_duration(video_path):
    """Get video duration in seconds"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    video.release()
    return duration


def get_video_info(video_path):
    """Get video metadata"""
    video = cv2.VideoCapture(video_path)
    
    info = {
        "fps": video.get(cv2.CAP_PROP_FPS),
        "frame_count": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(video.get(cv2.CAP_PROP_FOURCC)),
    }
    
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    
    video.release()
    return info


def is_valid_video(video_path):
    """Check if file is a valid video"""
    try:
        video = cv2.VideoCapture(video_path)
        ret, _ = video.read()
        video.release()
        return ret
    except:
        return False
