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
import signal
from training_resnet_functions import AgeGenderResNet

# parser = argparse.ArgumentParser(description="which cuda is used")
# parser.add_argument("input_number", type=int, help="An integer input number")
# parser.add_argument("-vd", "--visible_device", type=int, help="On which cuda is the model run", default=3)
# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.input_number)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model_weights_path = '/user/vhassle/u11216/psych_track/AgeSelf/age_classification_model_20_focal.pth'
model_weights_path = '/usr/users/vhassle/psych_track/AgeSelf/models/faces_a_g_1_0.02/age_gender_classification_model_final.pth'
model_age_name = model_weights_path.split("/")[-1].split(".")[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model_a_g = AgeGenderResNet()
model_a_g.load_state_dict(torch.load(model_weights_path))
model_a_g = model_a_g.to(device)
model_a_g.eval()

model_face_detection = load_Retinanet("/usr/users/vhassle/psych_track/Pytorch_Retinaface/Resnet50_Final.pth")
model_face_detection = model_face_detection.to(device)
model_face_detection.eval()

# model_face_detection = load_Retinanet("/user/vhassle/u11216/psych_track/Pytorch_Retinaface/Resnet50_Final.pth")
# model_face_detection = model_face_detection.to(device)
# model_face_detection.eval()


video_paths_prelim = glob.glob("/usr/users/vhassle/datasets/Wortschatzinsel/Neon_complete/Neon/*/2024_*.mp4")
video_paths_prelim = glob.glob("/usr/users/vhassle/datasets/Wortschatzinsel/all_videos/*.mp4")
video_paths_prelim = video_paths_prelim[1:3]
output_dir_short = "/usr/users/vhassle/model_outputs/outputs_AgeSelf_test" #phobos

# video_paths_prelim = ["/mnt/lustre-emmy-ssd/usr/u11216/data/wortschatzinsel/all_videos/*.mp4"]
# output_dir_short = "/mnt/lustre-emmy-ssd/usr/u11216/outputs"

output_dir = os.path.join(output_dir_short, model_age_name)
run_nr = "_r003"

os.makedirs(output_dir, exist_ok=True)
video_paths_prelim = sorted(video_paths_prelim)

video_paths = []
corrupted_paths = []
print("Checking for valid videos...")
def timeout_handler(signum, frame):
    raise TimeoutError
for video_path in tqdm(video_paths_prelim):
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)  # set a 2-second timeout
        de.VideoReader(video_path, ctx=de.cpu(0))
        signal.alarm(0)  # cancel the timeout
        video_paths.append(video_path)
    except TimeoutError:
        video_paths.append(video_path)  # append the path even if it timed out
        continue
    except: 
        base_name = os.path.basename(video_path)
        corrupted_path = os.path.join(os.path.dirname(video_path), f"corrupted_{base_name}")
        corrupted_paths.append(corrupted_path)
        continue

signal.alarm(0)   
with open(os.path.join(output_dir, "corrupted_paths.txt"), "w") as f:
    for path in corrupted_paths:
        f.write(path + "\n")

for video_path in tqdm(video_paths_prelim):
    try:
        de.VideoReader(video_path, ctx=de.cpu(0))
        video_paths.append(video_path)

    except: 
        base_name = os.path.basename(video_path)
        corrupted_path = os.path.join(os.path.dirname(video_path), f"corrupted_{base_name}")
        #os.rename(video_path, corrupted_path)
        continue


for video_path in tqdm(video_paths[0:10]):
    base_name = os.path.basename(video_path).split(".")[0]
    output_video_path = os.path.join(output_dir, base_name + run_nr + ".mp4")
    print(f"output_video_path: {output_video_path}")
    output_annotations_path = os.path.join(output_dir, base_name + run_nr + ".txt")
    if os.path.exists(output_annotations_path):
        continue

    process_video(video_path = video_path, model_a_g = model_a_g, model_face_detection= model_face_detection, 
                  output_video_path = output_video_path, output_annotations_path = output_annotations_path, image_size=150)
    
    