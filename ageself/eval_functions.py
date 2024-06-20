import pandas as pd
import numpy as np
#imports second part:
import cv2
import dlib

MINIMAL_SIZE = 20000
#helper 
def crop_image(image, annotation):
        annotation = [0 if x < 0 else int(x) for x in annotation]
        x, y, w, h = int(annotation[1]), int(annotation[2]), int(annotation[3]), int(annotation[4])
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image


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
import torchvision.transforms as transforms
import torch
class ResizeToMaxDim:
    def __init__(self, max_size):
        self.max_size = max_size
    
    def __call__(self, img):
        # Get current size
        h, w, _ = img.shape
        if w > h:
            new_w = self.max_size
            new_h = int(h * (self.max_size / w))
        else:
            new_h = self.max_size
            new_w = int(w * (self.max_size / h))
        
        # Resize image
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return resized_img
    
class PadToSquare:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill
    
    def __call__(self, img):
        h, w, _ = img.shape
        pad_w = (self.size - w) // 2
        pad_h = (self.size - h) // 2
        
        # Padding values for top, bottom, left, right
        padding = [(pad_h, self.size - h - pad_h), (pad_w, self.size - w - pad_w), (0, 0)]
        
        # Pad image
        padded_img = np.pad(img, padding, mode='constant', constant_values=self.fill)
        return padded_img

# image_size = 150
# transform_val = transforms.Compose([
#     ResizeToMaxDim(image_size),
#     PadToSquare(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


import torch.nn as nn
import torchvision
class AgeGenderResNet(nn.Module):
    def __init__(self, num_age_classes=3, num_gender_classes=2):
        super(AgeGenderResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.age_fc = nn.Linear(num_ftrs, num_age_classes)
        self.gender_fc = nn.Linear(num_ftrs, num_gender_classes)

    def forward(self, x):
        x = self.resnet(x)
        age_out = self.age_fc(x)
        gender_out = self.gender_fc(x)
        return age_out, gender_out

def estimate_age_gender_AgeSelf(image, annotations, specific_arguments):
    model = specific_arguments[0]
    for annotation in annotations:
        cropped_image = crop_image(image, annotation)
        # if cropped_image.shape[0]*cropped_image.shape[1] < MINIMAL_SIZE: #20000 for going_out, 200 for top
        #     continue


        # Define transformation
        image_size = 150
        transform = transforms.Compose([
            ResizeToMaxDim(image_size),
            PadToSquare(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load and preprocess the image
        input_image_before_cuda = transform(cropped_image)
        input_image = input_image_before_cuda.unsqueeze(0).to('cuda')
        # Make prediction
        with torch.no_grad():
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



        annotation.append(predict_gender)  # Gender
        annotation.append(predicted_age_group)  # Age