# This is the File that annotates the results of MOTIP with the according stuff and uses therefore stuff simmilar to what is done whith the faces think about merging longterm
# leftovers of FairFace and the other implementation are still here. One could run back to older git versions if it doesent work any longer I guess
from tqdm import tqdm, trange
import os
#imports second part:
import torch

import dlib
from ageself.eval_functions import estimate_age_gender_MiVolo, estimate_age_gender_FairFace, estimate_age_gender_AgeSelf, VideoDataset, save_annotated_video
from ageself.training_resnet_functions import load_age_gender_resnet

from mivolo.predictor import Predictor #age/gender estimation Model that is good but not available right now with usefull weights
import time
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

# Usage example

#video_name = "2024_05_04_10_57_26"
#video_name = "2024_05_19_10_51_24"
# video_name = "2024_05_04_14_31_45"

# video_path = f"/usr/users/vhassle/datasets/Wortschatzinsel/Neon_complete/Neon/{video_name.replace('_','-')}/{video_name}.mp4"
# annotation_path = f"/usr/users/vhassle/psych_track/MOTIP/outputs/Wortschatzinsel/Neon_test/detector/{video_name}.mp4.txt"
# #annotation_path = f"/usr/users/vhassle/psych_track/MOTIP/outputs/Wortschatzinsel/Neon_test/tracker/{video_name}.mp4.txt"

#top view 
view = "top"
annotation_path = "/usr/users/vhassle/psych_track/MOTIP/outputs/Wortschatzinsel/Neon_test/detector/2024-05-04 12-42-04.mkv.txt"
video_path = "/usr/users/vhassle/datasets/Wortschatzinsel/2024-05-04 12-42-04.mkv"

output_path = f'/usr/users/vhassle/psych_track/MOTIP/outputs/{os.path.basename(annotation_path).split(".")[0]}_Top_a_g.mp4'#_MiVOLO.mp4'


# estimate_age_gender_MiVolo
# Initialize Predictor
model_weights_path = '/usr/users/vhassle/psych_track/AgeSelf/models/body_a_g_1_0.02/body_a_g_classification_model_final.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


model_a_g = load_age_gender_resnet(model_weights_path)
specific_arguments = [model_a_g]


dataset = VideoDataset(video_path, view=view)
save_annotated_video(output_path, annotation_path, specific_arguments, estimate_age_gender_AgeSelf, age_gender_estimation=True)


# # estimate_age_gender_MiVolo
# # Initialize Predictor
# class Args:
#     def __init__(self):
#         self.detector_weights = "/usr/users/vhassle/psych_track/MiVOLO/models/yolov8x_person_face.pt"
#         self.checkpoint = "/usr/users/vhassle/psych_track/MiVOLO/models/mivolo_imbd.pth.tar"
#         self.with_persons = True
#         self.disable_faces = False
#         self.draw = False
#         self.device = "cuda"

# args = Args()
# predictor = Predictor(args, verbose=False)
# specific_arguments = [predictor]
# dataset = VideoDataset(video_path)

# dataset.save_annotated_video(output_path, annotation_path, "quatsch", specific_arguments, estimate_age_gender_MiVolo, age_gender_estimation=False)



# # estimate_age_gender_FairFace
# cnn_face_detector = dlib.cnn_face_detection_model_v1('/usr/users/vhassle/psych_track/FairFace/dlib_models/mmod_human_face_detector.dat')
# sp = dlib.shape_predictor('/usr/users/vhassle/psych_track/FairFace/dlib_models/shape_predictor_5_face_landmarks.dat')
# model_path = "/usr/users/vhassle/psych_track/FairFace/fair_face_models/res34_fair_align_multi_7_20190809.pt"

# trans = torchvision.transforms.Compose([
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                      std=[0.229, 0.224, 0.225])
# ])

# device = "cuda:0"
# model_fair_7 = torchvision.models.resnet34(weights=True)
# model_fair_7.fc = torch.nn.Linear(model_fair_7.fc.in_features, 18)
# model_fair_7.load_state_dict(torch.load(model_path))
# model_fair_7 = model_fair_7.to(device)
# model_fair_7.eval()

# specific_arguments = [cnn_face_detector, sp, model_fair_7, trans, device]
# dataset = VideoDataset(video_path)

# dataset.save_annotated_video(output_path, annotation_path, "non_specific_view", specific_arguments, estimate_age_gender_FairFace, age_gender_estimation=True)