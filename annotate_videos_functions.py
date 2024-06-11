import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
import decord
from torchvision import models, transforms
from pytorch_retinaface.detect import process_image, load_Retinanet #self created module self installed
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class ResizeToMaxDim:
    def __init__(self, max_size):
        self.max_size = max_size
    
    def __call__(self, img):
        # Get current size
        w, h = img.size
        if w > h:
            new_w = self.max_size
            new_h = int(h * (self.max_size / w))
        else:
            new_h = self.max_size
            new_w = int(w * (self.max_size / h))
        return img.resize((new_w, new_h), Image.LANCZOS)
    
class PadToSquare:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill
    
    def __call__(self, img):
        w, h = img.size
        pad_w = (self.size - w) // 2
        pad_h = (self.size - h) // 2
        padding = (pad_w, pad_h, self.size - w - pad_w, self.size - h - pad_h)
        return transforms.functional.pad(img, padding, fill=self.fill)


def get_annotations(image, model_a_g, model_face_detection ,index_frame = 0, image_size=150, 
                                   
                                   ):
    """
    Plots an image with estimated ages annotated on detected faces. and returns the annotations in MOT format.

    Parameters:
    image (Union[str, np.ndarray]): Path to the image or a NumPy array representing the image.
    model: The model used for age and gender estimation.
    image_size (int): The size to which the image should be resized.
    """
    # Define transformation
    transform = transforms.Compose([
        ResizeToMaxDim(image_size),
        PadToSquare(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the image
    if isinstance(image, str):
        frame_bgr = cv2.imread(image)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        frame_rgb = image
    else:
        raise ValueError("image must be a file path or a numpy array")

    # Detect faces
    faces = process_image(model_face_detection, frame_rgb)#frame should be rgb
    height, width, _ = frame_rgb.shape

    # Prepare for prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(index_frame)
    cv2.putText(frame_rgb, index_frame, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255), 2)
    annotations = []

    for face_key in faces.keys():
        face_area = faces[face_key]
        x1, y1, x2, y2 = face_area
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

        # Extract the face
        face = frame_rgb[y1:y2, x1:x2]

        # Convert the face to PIL image and apply transformation
        face_pil = Image.fromarray(face)
        input_image_before_cuda = transform(face_pil)
        input_image = input_image_before_cuda.unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            age_output, gender_output = model_a_g(input_image)
            #if len(output) #means that also gender is estimated
            # gender_output, age_output
            predicted_age_group = nn.Softmax(dim=1)(age_output)
            age_group = predicted_age_group.argmax(dim=1).item()
            predicted_age_text = f'{age_group}'

            gender = gender_output.argmax(dim=1).item()
            predicted_gender_text = f'{gender}'

        # Annotate the image using OpenCV
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_rgb, predicted_age_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame_rgb, predicted_gender_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Prepare annotation for MOT format
        annotation = [index_frame, face_key, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1, age_group]
        annotations.append(annotation)
    

    return frame_rgb, annotations

# write dataset
class VideoDataset(Dataset):
    def __init__(self, video_path):
        """
        Args: video_path (str): Path to the video file
        """
        # Initialize the VideoReader
        self.vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # Load video in CPU memory
        self.length = len(self.vr)  # Total number of frames
        self.video_size = self.vr[0].asnumpy().shape

    def __getitem__(self, idx):
        frame = self.vr[idx].asnumpy()
        return frame

    def __len__(self):
        return self.length

# Process video and save annotated video
def process_video(video_path, model_a_g, model_face_detection, output_video_path, output_annotations_path, image_size=448):
    dataset = VideoDataset(video_path)

    video_writer = None
    annotations = []

    for idx, frame in enumerate(tqdm(range(len(dataset)))):
        frame = dataset[idx]
        annotated_frame, frame_annotations = get_annotations(image = frame, model_a_g = model_a_g, model_face_detection=  model_face_detection,
                                                             index_frame = idx, image_size = image_size)

        # Initialize video writer
        if video_writer is None:
            height, width, _ = annotated_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

        video_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        for annotation in frame_annotations:
            annotation[0] = idx  # Set frame index
            annotations.append(annotation)

    video_writer.release()

    # Save annotations in MOT style
    with open(output_annotations_path, 'w') as f:
        for annotation in annotations:
            f.write(','.join(map(str, annotation)) + '\n')

    print(f"Annotated video saved to {output_video_path}")
    print(f"Annotations saved to {output_annotations_path}")