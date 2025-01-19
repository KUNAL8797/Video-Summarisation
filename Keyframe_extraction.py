import pickle
import cv2
import numpy as np
import os
from Kmeans_improvment import kmeans_silhouette
from save_keyframe import save_frames
from Redundancy import redundancy

# Define paths
scenes_path = "/Volumes/Evm Elite/key/data/shot_segmentation/trav2/kLxoNp-UchI_all_frames.npy"
features_path = "/Volumes/Evm Elite/key/data/shot_segmentation/cipi/kLxoNp-UchI_features.pkl"
video_path = "/Volumes/Evm Elite/key/Keyframe-Extraction-for-video-summarization/src/extraction/kLxoNp-UchI.mp4"
folder_path = "/Volumes/Evm Elite/key/data/key"
save_path = "/Volumes/Evm Elite/key/data/key"

# Print paths for verification
print("Scenes Path:", scenes_path)
print("Features Path:", features_path)
print("Video Path:", video_path)
print("Folder Path:", folder_path)
print("Save Path:", save_path)

# Check if the paths exist and are accessible
print("\nPath Validations:")
print(f"Scenes Path exists: {os.path.exists(scenes_path)}")
print(f"Features Path exists: {os.path.exists(features_path)}")
print(f"Video Path exists: {os.path.exists(video_path)}")
print(f"Folder Path exists: {os.path.exists(folder_path)}")
print(f"Save Path is writable: {os.access(save_path, os.W_OK)}")


import pickle
import cv2
import numpy as np
import os
from Kmeans_improvment import kmeans_silhouette
from save_keyframe import save_frames
from Redundancy import redundancy

def scen_keyframe_extraction(scenes_path, features_path, video_path, save_path, folder_path):
    try:
        # Read segmentation data (shot boundaries)
        number_list = []
        
        # Handle file reading errors due to encoding issues
        try:
            with open(scenes_path, 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                for line in lines:
                    print(f"Raw line from scenes file: {line}")  # Debugging print
                    numbers = line.strip().split(' ')
                    print(f"Extracted numbers: {numbers}")  # Debugging print
                    number_list.extend([int(number) for number in numbers])
        except UnicodeDecodeError:
            # If an encoding error occurs, try reading in binary mode
            with open(scenes_path, 'rb') as file:
                lines = file.readlines()
                for line in lines:
                    print(f"Raw line from scenes file (binary): {line}")  # Debugging print
                    numbers = line.strip().split(b' ')
                    print(f"Extracted numbers: {numbers}")  # Debugging print
                    number_list.extend([int(number) for number in numbers])

        print(f"Number list: {number_list}")  # Debugging print

        # Read features data (from pickle)
        with open(features_path, 'rb') as file:
            features = pickle.load(file)

        features = np.asarray(features)
        print(f"Features loaded. Shape: {features.shape}")  # Debugging print

        # Ensure that number_list is not empty
        if not number_list:
            print("Error: Number list is empty!")
            return

        # Initialize keyframe index list
        keyframe_index = []

        # Process each shot and perform keyframe extraction
        for i in range(0, len(number_list) - 1, 2):
            start = number_list[i]
            end = number_list[i + 1]

            print(f"Processing range: {start} to {end}")  # Debugging print

            # Validate range before slicing
            if start < 0 or end > len(features):
                print(f"Invalid range: {start}-{end}, skipping this shot.")
                continue

            # Extract the sub-feature set for the current shot
            sub_features = features[start:end]
            print(f"Sub-features shape: {sub_features.shape}")  # Debugging print

            if sub_features.shape[0] == 0:
                print(f"Skipping empty sub-features for range {start}-{end}")
                continue  # Skip empty sub-features

            # Perform KMeans clustering on the sub-features
            best_labels, best_centers, k, index = kmeans_silhouette(sub_features)
            print(f"KMeans indices: {index}")  # Debugging print

            # Adjust the final indices to match the overall video frame numbering
            final_index = [x + start for x in index]
            print(f"Final keyframe indices for range {start}-{end}: {final_index}")  # Debugging print

            # Apply redundancy filter to remove duplicate keyframes
            final_index = redundancy(video_path, final_index, 0.94)
            print(f"After redundancy filtering: {final_index}")  # Debugging print

            # Add the filtered final indices to the overall keyframe index list
            keyframe_index += final_index

        # Sort the final keyframe indices
        keyframe_index.sort()
        print(f"Final keyframe indices: {keyframe_index}")  # Debugging print

        # If no valid keyframes were extracted, exit early
        if not keyframe_index:
            print("No keyframes found to save!")
            return

        # Save the extracted keyframes
        print("Starting to save keyframes...")
        save_frames(keyframe_index, video_path, save_path, folder_path)
        print("Keyframes saved successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Call the function to start the extraction process
import pickle
import cv2
import numpy as np

from Kmeans_improvment import kmeans_silhouette
from save_keyframe import save_frames
from Redundancy import redundancy


def scen_keyframe_extraction(scenes_path, features_path, video_path, save_path, folder_path):
    # Get lens segmentation data
    number_list = []
    with open(scenes_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line)
            numbers = line.strip().split(' ')
            print(numbers)
            number_list.extend([int(number) for number in numbers])

    # Read inference data from local
    with open(features_path, 'rb') as file:
        features = pickle.load(file)

    features = np.asarray(features)
    # print(len(features))

    # Clustering at each shot to obtain keyframe sequence numbers
    keyframe_index = []
    for i in range(0, len(number_list) - 1, 2):
        start = number_list[i]
        end = number_list[i + 1]
        # print(start, end)
        sub_features = features[start:end]
        best_labels, best_centers, k, index = kmeans_silhouette(sub_features)
        # print(index)
        final_index = [x + start for x in index]
        # final_index.sort()
        # print("clustering：" + str(keyframe_index))
        # print(start, end)
        final_index = redundancy(video_path, final_index, 0.94)
        # print(final_index)
        keyframe_index += final_index
    keyframe_index.sort()
    print("final_index：" + str(keyframe_index))

    # save keyframe
    save_frames(keyframe_index, video_path, save_path, folder_path)



