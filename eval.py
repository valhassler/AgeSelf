#basically the same as above but now also gender and age estiamtions are made. If used in future clean it and merge it

from tqdm import tqdm, trange
import os
#imports second part:
import torch
import csv
import cv2
import decord
from torch.utils.data import Dataset
import torchvision

import dlib
from ageself.eval_functions import estimate_age_gender_MiVolo, estimate_age_gender_FairFace, estimate_age_gender_AgeSelf, AgeGenderResNet
from mivolo.predictor import Predictor
import time
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")




class VideoDataset(Dataset):
    def __init__(self, video_path, view="none"):
        """
        Args: video_path (str): Path to the video file
        """
        # Initialize the VideoReader
        self.vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # Load video in CPU memory
        self.view= view
        if view in ['top', 'coming_in','going_out']:
            self.x_th_frame = 100
        else:
            self.x_th_frame = 1
        self.length = int(len(self.vr)/self.x_th_frame)  # Total number of frames
        self.video_size = self.get_view(self.vr[0].asnumpy(), view = self.view).shape
        
        print(f"Video len: {self.length}")
    
    def load_annotations(self, annotation_path):
        annotations = {}
        with open(annotation_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                frame_idx = int(row[0]) - 1  # Assuming frame indices in the file start from 1
                annotation = list(map(float, row[1:]))  # Convert the rest of the row to floats
                if frame_idx not in annotations:
                    annotations[frame_idx] = []
                annotations[frame_idx].append(annotation)
        return annotations
    
    def get_view(self, np_array, view="all"):
        entire_image = np_array
        if view == 'top':
            image = entire_image[0:540, 62:892]  # Crop from top view
        elif view == 'coming_in':
            image = entire_image[540:1500, 0:540]
        elif view == 'going_out':
            image = entire_image[540:1500, 540:1080]
        else:
            image = entire_image
        return image


    def draw_annotations(self, frame, annotations, frame_number):
        """
        All annotations for bbox and age gender

        """
        cv2.putText(frame, f"{frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        for annotation in annotations:
            obj_id, x, y, w, h,confidence, gender, age = int(annotation[0]), annotation[1], annotation[2], annotation[3], annotation[4],annotation[5], annotation[-2], annotation[-1]
            top_left = (int(x), int(y))
            bottom_right = (int(x + w), int(y + h))
            gender = gender if gender !=-1 else None
            age = age if age !=-1 else None
            if confidence < 0.5:
                continue
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f'conf: {round(confidence, 2)}', (int(x), int(y +20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
            cv2.putText(frame, f'ID: {obj_id}', (int(x), int(y - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            cv2.putText(frame, f'Age: {age}', (int(x), int(y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
            cv2.putText(frame, f'Gender: {gender}', (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
        return frame
    def attempt_execution(self, vr, idx, retries=100, delay=0.1):
        for attempt in range(retries):
            try:
                #frame = vr[idx].asnumpy()
                frame = self.__getitem__(idx)
                #print(f"Attempt {attempt + 1}: Success")
                # You can return or process the frame if needed
                return frame
            except Exception as e:
                print(f"Attempt {attempt + 1}: Failed with error {e}")
                vr[0].asnumpy()
                time.sleep(delay)  # Wait for the specified delay before retrying

        print("All attempts failed.")
        return "frame_failed"
    def save_annotated_video(self, output_path, annotation_path, predictor_age_gender,age_gender_est_func, age_gender_estimation=False):
        """
        ## Args:
        - output_path (str): Path to save the annotated video
        - annotation_path (str): Path to the annotation file
        - age_gender_est_func: function that uses image, annotations and predictor, estimates age and gender
        and puts it in the annotations to the bounding box as additional information
        - predictor:  Initilized model for the age and gender prediciton
        """
        annotations = self.load_annotations(annotation_path)
        age_gender_basepath = "/".join(os.path.dirname(annotation_path).split("/")[:-1]) + "/tracker_age_gender"
        os.makedirs(age_gender_basepath, exist_ok=True)
        annotation_with_age_gender_path = os.path.join(age_gender_basepath,os.path.basename(annotation_path).split(".")[0] + "_age_gender.txt")
        
        height, width = self.video_size[0], self.video_size[1]
        
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        
        for idx in tqdm(range(self.length)):
            frame = self.attempt_execution(self.vr, idx, retries=3, delay=0.05)
            if  isinstance(frame, str):
                continue
            selected_view = frame
                
            frame_annotations = annotations.get(idx, [])
            if idx < 150000 and age_gender_estimation:
                age_gender_est_func(selected_view, frame_annotations, predictor_age_gender) #frame annotations are appended in this function, and also the frame is annotated
            annotated_frame = self.draw_annotations(selected_view, frame_annotations, idx)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            if age_gender_estimation is not True:
                continue
            # Write the annotations to a file
            if idx == 0 and os.path.exists(annotation_with_age_gender_path):
                os.remove(annotation_with_age_gender_path)
            if len(annotation_with_age_gender_path) > 0:
                for annotation in frame_annotations:
                    indices = [0, 1, 2, 3, 4, -2, -1]
                    content_to_write = str(idx) + " " +' '.join(str(annotation[i]) for i in indices)
                    with open(annotation_with_age_gender_path, 'a') as file:
                        file.write(content_to_write + '\n')
        out.release()

    def __getitem__(self, idx):
        frame = self.vr[idx*self.x_th_frame]
        frame = frame.asnumpy()
        frame = self.get_view(frame, view=self.view)
        return frame

    def __len__(self):
        return self.length

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
model_a_g = AgeGenderResNet()
model_a_g.load_state_dict(torch.load(model_weights_path))
model_a_g = model_a_g.to(device)
model_a_g.eval()


specific_arguments = [model_a_g]
dataset = VideoDataset(video_path, view=view)
dataset.save_annotated_video(output_path, annotation_path, specific_arguments, estimate_age_gender_AgeSelf, age_gender_estimation=True)



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

