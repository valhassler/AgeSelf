import glob
import torch
import torch.nn as nn
import torchvision.models as models
import os
import decord as de
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp

from annotate_videos_functions import process_video
from pytorch_retinaface.detect import process_image, load_Retinanet  # self created module self installed

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def initialize_models():
    model_weights_path = '/user/vhassle/u11216/psych_track/AgeSelf/age_classification_model_final_focal.pth'
    model_age_name = model_weights_path.split("/")[-1].split(".")[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_age = models.resnet50(pretrained=False)
    num_ftrs = model_age.fc.in_features
    model_age.fc = nn.Linear(num_ftrs, 3)
    model_age.load_state_dict(torch.load(model_weights_path))
    model_age = model_age.to(device)
    model_age.eval()

    model_face_detection = load_Retinanet("/user/vhassle/u11216/psych_track/Pytorch_Retinaface/Resnet50_Final.pth")
    model_face_detection = model_face_detection.to(device)
    model_face_detection.eval()

    return model_age, model_face_detection, device

# Define paths
video_paths_prelim = glob.glob("/mnt/lustre-emmy-ssd/usr/u11216/data/wortschatzinsel/all_videos/*.mp4")
output_dir_short = "/mnt/lustre-emmy-hdd/usr/u11216/outputs"
model_age_name = 'age_classification_model_final_focal'
output_dir = os.path.join(output_dir_short, model_age_name)
run_nr = "_r002"

os.makedirs(output_dir, exist_ok=True)
video_paths_prelim = sorted(video_paths_prelim)

video_paths = []
corrupted_paths = []
print("Checking for valid videos...")

# Check for corrupted files
for video_path in tqdm(video_paths_prelim):
    try:
        de.VideoReader(video_path, ctx=de.cpu(0))
        video_paths.append(video_path)
    except:
        base_name = os.path.basename(video_path)
        corrupted_path = os.path.join(os.path.dirname(video_path), f"corrupted_{base_name}")
        corrupted_paths.append(corrupted_path)
        continue

# Save corrupted paths to a file
with open(os.path.join(output_dir, "corrupted_paths.txt"), "w") as f:
    for path in corrupted_paths:
        f.write(path + "\n")

def process_chunk(video_chunk):
    model_age, model_face_detection, device = initialize_models()
    for video_path in tqdm(video_chunk):
        print(f"Processing {video_path} in process {os.getpid()}")
        base_name = os.path.basename(video_path).split(".")[0]
        output_video_path = os.path.join(output_dir, base_name + run_nr + ".mp4")
        output_annotations_path = os.path.join(output_dir, base_name + run_nr + ".txt")
        if os.path.exists(output_annotations_path):
            continue

        process_video(video_path=video_path, model_age=model_age, model_face_detection=model_face_detection,
                      output_video_path=output_video_path, output_annotations_path=output_annotations_path,
                      classification=True, image_size=150)

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)  # Set the multiprocessing start method to 'spawn'
    num_processes = 15
    chunk_size = len(video_paths) // num_processes
    chunks = [video_paths[i:i + chunk_size] for i in range(0, len(video_paths), chunk_size)]

    # Use concurrent futures to process chunks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
