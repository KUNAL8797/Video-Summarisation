import os
import gc
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch
from torch.utils.data import DataLoader
import pickle  # Import the pickle module

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Path for input videos and output features
input_path = "/Volumes/Evm Elite/FINAL/terserv2/TransNetV2/inference/kLxoNp-UchI.mp4"  # Can be a directory or a single file
output_dir = "/Volumes/Evm Elite/FINAL/terserv2/data/shot_segmentation/cipi"
os.makedirs(output_dir, exist_ok=True)

# Function to process a single video
def process_video(video_path, output_dir):
    video_name = os.path.basename(video_path)  # Extract the video name
    print(f"Processing {video_name} for features...")

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(Image.fromarray(frame))
    finally:
        cap.release()
        gc.collect()  # Explicit garbage collection

    # Process frames in batches
    batch_size = 32
    dataloader = DataLoader(frames, batch_size=batch_size, collate_fn=lambda x: processor(images=x, return_tensors="pt", padding=True))
    
    features_list = []
    for batch in dataloader:
        with torch.no_grad():
            batch_features = model.get_image_features(**batch)
        features_list.append(batch_features)

    # Concatenate all features
    features = torch.cat(features_list)

    # Save features in .pkl format
    output_path = os.path.join(output_dir, video_name.replace('.mp4', '_features.pkl'))
    with open(output_path, 'wb') as f:
        pickle.dump(features.detach().numpy(), f)
    print(f"Features saved to {output_path}")

    # Free memory
    frames.clear()

# Determine if input_path is a directory or a single file
if os.path.isdir(input_path):
    # Process all videos in the directory
    for video_name in os.listdir(input_path):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(input_path, video_name)
            process_video(video_path, output_dir)
elif os.path.isfile(input_path) and input_path.endswith('.mp4'):
    # Process a single video file
    process_video(input_path, output_dir)
else:
    print(f"Invalid input path: {input_path}")