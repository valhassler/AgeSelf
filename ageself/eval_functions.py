import pandas as pd
import numpy as np
#imports second part:
import cv2
import dlib

from torch import no_grad
import torch.nn as nn
import torchvision
import os 
import decord
import time
import csv
from torch.utils.data import Dataset
from tqdm import tqdm
import PIL


from ageself.training_resnet_functions import get_val_transform
MINIMAL_SIZE = 20000
#helper 
def crop_image(image, annotation):
        annotation = [0 if x < 0 else int(x) for x in annotation]
        x, y, w, h = int(annotation[1]), int(annotation[2]), int(annotation[3]), int(annotation[4])
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image

def load_annotations(annotation_path):
    """
    args: annotation_path (str): Path to the annotation file
    returns: annotations (dict): Dictionary containing annotations for each frame
    explanation: The Index is set to start from 0, all elements are converted to float

    """
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

def draw_annotations(frame, annotations, frame_number):
    """
    All annotations for bbox and age gender

    """
    cv2.putText(frame, f"{frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    if annotations is None:
        return frame
    
    for annotation in annotations:
        #that is the expected format
        obj_id, x, y, w, h,confidence, gender, age = int(annotation[0]), annotation[1], annotation[2], annotation[3], annotation[4],annotation[5], annotation[-2], annotation[-1]
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        gender = gender if gender !=-1 else None
        age = age if age !=-1 else None
        if confidence < 0.5:
            continue
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        # cv2.putText(frame, f'conf: {round(confidence, 2)}', (int(x), int(y +20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
        # cv2.putText(frame, f'ID: {obj_id}', (int(x), int(y - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {age}', (int(x), int(y + 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
        cv2.putText(frame, f'Gender: {gender}', (int(x), int(y + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
    return frame

def attempt_execution(vr, idx, retries=100, delay=0.1):
    """
    use: tries to get the frame from the video reader, if it fails it retries it
    """
    for attempt in range(retries):
        try:
            #frame = vr[idx].asnumpy()
            frame = vr[idx]
            #print(f"Attempt {attempt + 1}: Success")
            # You can return or process the frame if needed
            return frame
        except Exception as e:
            print(f"Attempt {attempt + 1}: Failed with error {e}")
            vr[0].asnumpy()
            time.sleep(delay)  # Wait for the specified delay before retrying

    print("All attempts failed.")
    return "frame_failed"


def save_annotated_video(video, output_path, annotation_path, predictor_age_gender,age_gender_est_func, age_gender_estimation=False):
    """
    ## Args:
    - video: Video initialized in a way that one can video[idx] to get the frame
    - output_path (str): Path to save the annotated video
    - annotation_path (str): Path to the annotation file
    - age_gender_est_func: function that uses image, annotations and predictor, estimates age and gender
    and puts it in the annotations to the bounding box as additional information
    - predictor:  Initilized model for the age and gender prediciton
    """
    annotations = load_annotations(annotation_path)
    age_gender_basepath = "/".join(os.path.dirname(annotation_path).split("/")[:-1]) + "/tracker_age_gender"
    os.makedirs(age_gender_basepath, exist_ok=True)
    annotation_with_age_gender_path = os.path.join(age_gender_basepath,os.path.basename(annotation_path).split(".")[0] + "_age_gender.txt")
    
    height, width = video[0].shape[:2]
    video_length = len(video)

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    
    for idx in tqdm(range(video_length)):
        frame = attempt_execution(video, idx, retries=3, delay=0.05)
        if  isinstance(frame, str):
            continue
        selected_view = frame
        frame_annotations = annotations.get(idx, [])


        if age_gender_estimation:
            frame_annotations = age_gender_est_func(selected_view, frame_annotations, predictor_age_gender) #frame annotations are updated in this function

        annotated_frame = draw_annotations(selected_view, frame_annotations, idx)
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        if age_gender_estimation is not True:
            continue
        # Write the annotations to a file

        if idx == 0 and os.path.exists(annotation_with_age_gender_path):
            os.remove(annotation_with_age_gender_path)
        if len(annotation_with_age_gender_path) > 0:
            if frame_annotations is None:
                continue
            for annotation in frame_annotations:
                indices = [0, 1, 2, 3, 4, -2, -1]
                content_to_write = str(idx) + " " +' '.join(str(annotation[i]) for i in indices)
                with open(annotation_with_age_gender_path, 'a') as file:
                    file.write(content_to_write + '\n')
    out.release()


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
    

    #use this as function in annotate video may be but also dont produce all the annotated videos I now know how it works, may be subdivide it also that not annotation and video annotation is done in the same step but
    #probably it does not reall ymatters
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

    def __getitem__(self, idx):
        frame = self.vr[idx*self.x_th_frame]
        frame = frame.asnumpy()
        frame = self.get_view(frame, view=self.view)
        return frame

    def __len__(self):
        return self.length
    


# FairFace
def reverse_resized_rect(rect,resize_ratio):
    l = int(rect.left() / resize_ratio)
    t = int(rect.top() / resize_ratio)
    r = int(rect.right() / resize_ratio)
    b = int(rect.bottom() / resize_ratio)
    new_rect = dlib.rectangle(l,t,r,b)
    
    return [l,t,r,b] , new_rect

def resize_image(img, default_max_size=800):
    old_height, old_width, _ = img.shape
    if old_width > old_height:
        resize_ratio = default_max_size / old_width
        new_width, new_height = default_max_size, int(old_height * resize_ratio)
    else:
        resize_ratio = default_max_size / old_height
        new_width, new_height =  int(old_width * resize_ratio), default_max_size
    #img = dlib.resize_image(img, cols=new_width, rows=new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return img, resize_ratio

def extract_faces(img, cnn_face_detector, sp):
    """
    Extracts faces from an image using dlib's cnn_face_detector and shape_predictor
    param img: image to extract faces from
    return: list of extracted faces and their corresponding bboxes
    """
    rects = []
    img, resize_ratio = resize_image(img)
    dets = cnn_face_detector(img, 1) #takes the longest
    num_faces = len(dets)
    faces = dlib.full_object_detections()

    for detection in dets:
        rect = detection.rect
        faces.append(sp(img, rect))
        rect_tpl ,rect_in_origin = reverse_resized_rect(rect,resize_ratio)
        rects.append(rect_in_origin)
    # seems to extract the faces and size them to 300x300
    if len(faces) > 0:
        faces_image = dlib.get_face_chips(img, faces, size=300, padding = 0.25) #predefined
        return faces_image, rects
    else:
        return [], []

def softmax_numpy(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.sum(exps)


def estimate_age_gender_FairFace(image, annotations, specific_arguments):
    cnn_face_detector, sp, model_g_a, trans, device = specific_arguments
    for annotation in annotations:
        cropped_image = crop_image(image, annotation)
        if cropped_image.shape[0]*cropped_image.shape[1] < MINIMAL_SIZE: #20000 for going_out, 200 for top
            continue
        faces_image, rects = extract_faces(cropped_image, cnn_face_detector, sp)
        # Now prediction of the images
        #zoom on one image
        observations = []
        for i, image in enumerate(faces_image):
            image = trans(image)
            image = image.view(1, 3, 224, 224) 
            image = image.to(device)
            outputs = model_g_a(image)

            outputs = model_g_a(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)
            ## Postprocessing 
            #race_outputs = outputs[:7]
            gender_outputs = outputs[7:9]
            age_outputs = outputs[9:18]

            #race_score = softmax_numpy(race_outputs)
            gender_score = softmax_numpy(gender_outputs)
            age_score = softmax_numpy(age_outputs)

            gender_pred = np.argmax(gender_score)
            age_pred = np.argmax(age_score)
            observations.append([gender_pred, age_pred,
                                gender_score, age_score])

        if len(observations) == 0:
            return
        else:
            result = pd.DataFrame(observations)
            result.columns = ['gender_preds','age_preds',
                                'gender_scores','age_scores']
            #bboxes
            # Mapping for gender predictions
            #in case of doubt take the older one
            gender_mapping = {0: 'Male', 1: 'Female'}
            age_mapping = {
                0: '0-2', 1: '3-9', 2: '10-19', 3: '20-29',
                4: '30-39', 5: '40-49', 6: '50-59', 7: '60-69', 8: '70+'}
            result['gender_preds'] = result['gender_preds'].map(gender_mapping)
            result['age_preds'] = result['age_preds'].map(age_mapping)
        
            annotation.append(result['gender_preds'][0]) 
            annotation.append(result['age_preds'][0])
# MiVOLO
def estimate_age_gender_MiVolo(image, annotations, specific_arguments):
    predictor = specific_arguments[0]
    for annotation in annotations:
        cropped_image = crop_image(image, annotation)
        if cropped_image.shape[0]*cropped_image.shape[1] < MINIMAL_SIZE: #20000 for going_out, 200 for top
            continue

        detected_objects, _ = predictor.recognize(cropped_image)
        if detected_objects.n_persons == 0:
            annotation.append(None)  # No gender
            annotation.append(None)  # No age
        else:
            annotation.append(detected_objects.genders[0])  # Gender
            annotation.append(np.mean(detected_objects.ages))  # Age
#AgeSelf


def estimate_age_gender_AgeSelf(image, annotations, specific_arguments):
    model = specific_arguments[0]
    for annotation in annotations:
        cropped_image = crop_image(image, annotation)
        # if cropped_image.shape[0]*cropped_image.shape[1] < MINIMAL_SIZE: #20000 for going_out, 200 for top
        #     continue


        # Define transformation
        image_size = 450
        transform = get_val_transform(image_size)

        # Loadp and preprocess the image#
        cropped_image = PIL.Image.fromarray(cropped_image)
        input_image_before_cuda = transform(cropped_image)
        input_image = input_image_before_cuda.unsqueeze(0).to('cuda')
        # Make prediction
        with no_grad():
            output = model(input_image)
            predicted_age_group, predict_gender = np.argmax(output[0].cpu().numpy()), np.argmax(output[1].cpu().numpy())
        # age_group_mapping = {
        #     0: '0-2',
        #     1: '3-13',
        #     2: '18-99',
        # }
        # predicted_age_group = age_group_mapping.get(predicted_age_group, 'Unknown')
        predicted_age_group = predicted_age_group
        predict_gender_final = "f" if predict_gender == 0 else "m" 

        annotation.append(predict_gender_final)  # Gender
        annotation.append(predicted_age_group)  # Age
    return annotations