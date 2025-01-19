import os
import cv2

def extract_keyfr(video_path, base_folder):
    default_keyframe_indices = [80, 230, 655, 712, 893, 964, 1230, 1950, 2070, 2900]
    video_filename_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Searching for folder related to: {video_filename_without_ext}")
    try:
        related_folder = None
        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            if os.path.isdir(item_path) and video_filename_without_ext in item:
                related_folder = item_path
                break
        if not related_folder:
            print(f"No folder found containing '{video_filename_without_ext}'")
            print("Using default keyframe indices.")
            return default_keyframe_indices
        folder_files = os.listdir(related_folder)
        keyframe = [os.path.splitext(file)[0] for file in folder_files if file.endswith('.jpg')]
        if not keyframe:
            return default_keyframe_indices
        for file in keyframe:
            print(file)
        return keyframe
    except Exception as e:
        return default_keyframe_indices

def save_frames(keyframe_indexes, video_path, save_path, folder_name):
    """
    Save specific frames from the video based on given indices
    :param keyframe_indexes: List of frame indices to save
    :param video_path: Path to the video file
    :param save_path: Directory to save extracted frames
    :param folder_name: Name of the folder to save frames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return

    # Create a folder path for saving images
    folder_path = os.path.join(save_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

    # Convert keyframe_indexes to integers
    keyframe_indexes = [int(index) for index in keyframe_indexes]

    # Print the keyframe indices for reference
    print("Keyframe Indices:", keyframe_indexes)

    # Initialize the current frame number
    current_index = 0

    # Start reading frames from the video
    print("Starting to save frames...")
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print(f"End of video or error reading frame at index {current_index}.")
            break

        # Save the frame if it is a keyframe
        if current_index in keyframe_indexes:
            file_name = f"{current_index}.jpg"
            file_path = os.path.join(folder_path, file_name)
            success = cv2.imwrite(file_path, frame)
            print(f"Attempting to save frame {current_index} as {file_name}: {'Success' if success else 'Failed'}")
        current_index += 1
    # Release resources
    cap.release()
    print("Completed saving frames.")

def main():
    # Paths to the video and output directories
    video_path = "/Volumes/Evm Elite/key/Keyframe-Extraction-for-video-summarization/src/extraction/kLxoNp-UchI.mp4"
    folder_name = "keyframes"  # folder to be saved.
    save_path = "/Volumes/Evm Elite/key/data/key"
    base_folder = "/Volumes/Evm Elite/key/data/Keyframe"
    keyframe_indexes = extract_keyfr(video_path, base_folder)
    # Verify paths and print details
    print(f"Video Path: {video_path}")
    print(f"Base Search Folder: {base_folder}")
    print(f"Save Path: {save_path}")
    print(f"Folder Name: {folder_name}")

    # Run the function to save frames
    save_frames(keyframe_indexes, video_path, save_path, folder_name)

if __name__ == "__main__":
    main()