from annotate_videos_functions import process_video
from pytorch_retinaface.detect import process_image, load_Retinanet #self created module self installed

import glob
import torch
import torch.nn as nn
import torchvision.models as models
import os
import argparse
import decord as de
from tqdm import tqdm
# parser = argparse.ArgumentParser(description="which cuda is used")
# parser.add_argument("input_number", type=int, help="An integer input number")
# parser.add_argument("-vd", "--visible_device", type=int, help="On which cuda is the model run", default=3)
# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.input_number)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_weights_path = '/usr/users/vhassle/psych_track/AgeSelf/models/age_classification_model_15_focal_pad.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model_age = models.resnet50(pretrained=False)
num_ftrs = model_age.fc.in_features
model_age.fc = nn.Linear(num_ftrs, 3)
model_age.load_state_dict(torch.load(model_weights_path))
model_age = model_age.to(device)
# torch.compile(model_age)
model_age.eval()

model_face_detection = load_Retinanet("/usr/users/vhassle/psych_track/Pytorch_Retinaface/Resnet50_Final.pth")
model_face_detection = model_face_detection.to(device)
# torch.compile(model_face_detection)
model_face_detection.eval()


# video_paths_prelim = glob.glob("/usr/users/vhassle/datasets/Wortschatzinsel/Neon_complete/Neon/*/2024_*.mp4")
video_paths_prelim = ["/usr/users/vhassle/datasets/Wortschatzinsel/Neon/test2.mp4"]
output_dir = "/usr/users/vhassle/psych_track/AgeSelf/outputs"
run_nr = "_r001"



os.makedirs(output_dir, exist_ok=True)
video_paths_prelim = sorted(video_paths_prelim)

video_paths = []
print("Checking for valid videos...")
for video_path in tqdm(video_paths_prelim):
    try:
        de.VideoReader(video_path, ctx=de.cpu(0))
        video_paths.append(video_path)

    except: 
        base_name = os.path.basename(video_path)
        corrupted_path = os.path.join(os.path.dirname(video_path), f"corrupted_{base_name}")
        os.rename(video_path, corrupted_path)
        continue


for video_path in tqdm(video_paths[0:10]):
    print(video_path)
    base_name = os.path.basename(video_path).split(".")[0]
    output_video_path = os.path.join(output_dir, base_name + run_nr + ".mp4")
    output_annotations_path = os.path.join(output_dir, base_name + run_nr + ".txt")
    if os.path.exists(output_annotations_path):
        continue

    process_video(video_path = video_path, model_age = model_age, model_face_detection= model_face_detection, 
                  output_video_path = output_video_path, output_annotations_path = output_annotations_path, 
                  classification=True, image_size=150)
    
    