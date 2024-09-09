
#Annotates a video with a_g using the face deteciton model and outputs then the video as mp4 and a txt in which the stuff is saved in a kinda MOT format
from ageself.annotate_videos_functions import process_video
from ageself.training_resnet_functions import load_age_gender_resnet
from pytorch_retinaface.detect import load_Retinanet #self created module self installed

import glob
import torch
import os
import decord as de 
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model_weights_path = '/usr/users/vhassle/psych_track/AgeSelf/models/faces_a_g_img_size_150_rot_90/age_gender_classification_model_final.pth'
model_age_name = model_weights_path.split("/")[-2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model_a_g = load_age_gender_resnet(model_weights_path)
model_face_detection = load_Retinanet("/usr/users/vhassle/psych_track/Pytorch_Retinaface/Resnet50_Final.pth")

video_paths_prelim = glob.glob("/usr/users/vhassle/datasets/Wortschatzinsel/all_videos/*.mp4")

output_dir_short = "/usr/users/vhassle/model_outputs/outputs_AgeSelf" #phobos
output_dir = os.path.join(output_dir_short, model_age_name)
run_nr = "_r003"

os.makedirs(output_dir, exist_ok=True)
# video_paths_prelim = sorted(video_paths_prelim)

# video_paths = []
# corrupted_paths = []
# print("Checking for valid videos...")
# for video_path in tqdm(video_paths_prelim):
#     try:
#         #de.VideoReader(video_path, ctx=de.cpu(0))
#         video_paths.append(video_path)
#     except: 
#         base_name = os.path.basename(video_path)
#         corrupted_path = os.path.join(os.path.dirname(video_path), f"corrupted_{base_name}")
#         corrupted_paths.append(corrupted_path)
#         continue

# with open(os.path.join(output_dir, "corrupted_paths.txt"), "w") as f:
#     for path in corrupted_paths:
#         f.write(path + "\n")

config = {"view":"top"}


video_paths = ["/usr/users/vhassle/datasets/Wortschatzinsel/2024-05-04 12-42-04.mkv"]
#Now annotate the videos required
for video_path in tqdm(video_paths[0:1]):
    base_name = os.path.basename(video_path).split(".")[0]
    output_video_path = os.path.join(output_dir, base_name + run_nr + ".mp4")
    print(f"output_video_path: {output_video_path}")
    output_annotations_path = os.path.join(output_dir, base_name + run_nr + ".txt")
    #if os.path.exists(output_annotations_path):
    #    continue

    process_video(video_path = video_path, model_a_g = model_a_g, model_face_detection= model_face_detection, 
                  output_annotations_path = output_annotations_path, output_video_path = "/usr/users/vhassle/video_eg.mp4", image_size=150, view = config["view"])