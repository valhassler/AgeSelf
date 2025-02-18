import glob
import torch
import torch.nn as nn
import torchvision.models as models
import os
import decord as de
from tqdm import tqdm

from ageself.annotate_videos_functions import process_video
from pytorch_retinaface.detect import process_image, load_Retinanet  # custom module
from ageself.training_resnet_functions import load_age_gender_resnet

# Base paths
user_base_path = "/mnt/lustre-emmy-hdd/usr/u11216"
model_weight_path = os.path.join(user_base_path,"psych_track/AgeSelf/age_gender_combined_training.pth") 

def initialize_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_a_g = load_age_gender_resnet(model_weight_path)
    model_face_detection = load_Retinanet(
        os.path.join(user_base_path,"psych_track/Pytorch_Retinaface/Resnet50_Final.pth")
    )
    return model_a_g, model_face_detection, device

# Define paths
video_paths_prelim = glob.glob(
    os.path.join(user_base_path,"datasets/Wortschatzinsel/eye_tracking/head_mounted_data/videos/*.mp4")
)
output_dir_short = os.path.join(user_base_path,"outputs")
output_dir = os.path.join(
    output_dir_short, 
    model_weight_path.split("/")[-1].split(".")[0] + "_r02_25v2"
)
run_nr = "_r02_25v2"

os.makedirs(output_dir, exist_ok=True)
video_paths_prelim = sorted(video_paths_prelim)

video_paths = []
corrupted_paths = []

print("Checking for valid videos...")

# Check for corrupted files
for video_path in tqdm(video_paths_prelim):
    try:
        # If decord can open the file, it is valid.
        de.VideoReader(video_path, ctx=de.cpu(0))
        video_paths.append(video_path)
    except:
        base_name = os.path.basename(video_path)
        corrupted_path = os.path.join(os.path.dirname(video_path), f"corrupted_{base_name}")
        corrupted_paths.append(corrupted_path)

# Save corrupted paths to a file
with open(os.path.join(output_dir, "corrupted_paths.txt"), "w") as f:
    for path in corrupted_paths:
        f.write(path + "\n")

def process_all_videos(video_list, model_a_g, model_face_detection, device):
    """Process each video sequentially."""
    for video_path in tqdm(video_list, desc="Processing videos"):
        print(f"Processing {video_path}")
        base_name = os.path.basename(video_path).split(".")[0]
        output_video_path = os.path.join(output_dir, base_name + run_nr + ".mp4")
        output_annotations_path = os.path.join(output_dir, base_name + run_nr + ".txt")

        # Skip if output video or annotation already exists
        if os.path.exists(output_video_path):
            print(f"Output video {output_video_path} already exists. Skipping {video_path}.")
            continue
        if os.path.exists(output_annotations_path):
            print(f"Output annotation {output_annotations_path} already exists. Skipping {video_path}.")
            continue

        # Process the video with your custom function
        process_video(
            video_path=video_path,
            model_a_g=model_a_g,
            model_face_detection=model_face_detection,
            output_annotations_path=output_annotations_path,
            output_video_path=output_video_path,
            image_size=150
        )

if __name__ == "__main__":
    # Initialize models once
    model_a_g, model_face_detection, device = initialize_models()

    # Process all videos in a single loop
    process_all_videos(video_paths, model_a_g, model_face_detection, device)
