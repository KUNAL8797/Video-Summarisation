import os
import numpy as np
from transnetv2 import TransNetV2

# Initialize TransNetV2 model
model = TransNetV2()

# Directories for input videos and output shot segmentation
input_dir = "/Volumes/Evm Elite/FINAL/terserv2/TransNetV2/inference/kLxoNp-UchI.mp4"  # Directory containing input videos
output_dir = "/Volumes/Evm Elite/FINAL/terserv2/data/shot_segmentation/trav2"  # Directory to save segmentation results
os.makedirs(output_dir, exist_ok=True)

# Function to process a single video
def process_video(video_name, video_path):
    print(f"Processing {video_name}...")
    
    # Predict shot boundaries using the TransNetV2 model
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)

    # Save results for single frame predictions
    output_path_single = os.path.join(output_dir, video_name.replace('.mp4', '_single_frame.npy'))
    #np.save(output_path_single, single_frame_predictions)

    # Save results for all frame predictions
    output_path_all = os.path.join(output_dir, video_name.replace('.mp4', '_all_frames.npy'))
    np.save(output_path_all, all_frame_predictions)
    print(f"All frame predictions saved to {output_path_all}")

    # Optional: Save video frames for inspection
    output_path_frames = os.path.join(output_dir, video_name.replace('.mp4', '_frames.npy'))
    #np.save(output_path_frames, video_frames)

# Check if input is a directory, a single file, or a dictionary
if os.path.exists(input_dir) and os.path.isdir(input_dir):
    # Process all videos in the directory
    for video_name in os.listdir(input_dir):
        if video_name.endswith('.mp4'):  # Process only MP4 video files
            video_path = os.path.join(input_dir, video_name)
            process_video(video_name, video_path)
elif isinstance(input_dir, dict):  # If it's a dictionary
    for video_name, video_path in input_dir.items():
        if os.path.exists(video_path) and video_path.endswith('.mp4'):
            process_video(video_name, video_path)
elif os.path.isfile(input_dir) and input_dir.endswith('.mp4'):  # Single video file
    video_name = os.path.basename(input_dir)
    process_video(video_name, input_dir)
else:
    print(f"Error: Invalid input '{input_dir}'! It must be a directory, a single video file, or a dictionary.")