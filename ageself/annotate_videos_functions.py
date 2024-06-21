import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
import decord
from pytorch_retinaface.detect import process_image #self created module self installed. 
from ageself.training_resnet_functions import get_val_transform


def detect_faces_and_annotate_initial(frame_rgb, model_face_detection, index_frame=0):
    """
    Detects faces in the image and prepares initial annotations.

    Parameters:
    image : NumPy array representing the image.
    model_face_detection: The model used for face detection.
    index_frame (int): The index of the frame being processed.

    Returns:
    Tuple: A tuple containing the RGB image with frame index annotated and a list of initial annotations.
    """
    # Check the image
    if not isinstance(frame_rgb, np.ndarray):
        ValueError("image must be a file path or a numpy array")
    # Detect faces
    faces = process_image(model_face_detection, frame_rgb) # Ensure frame is RGB
    height, width, _ = frame_rgb.shape

    # Annotate frame index
    cv2.putText(frame_rgb, str(index_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Prepare initial annotations
    initial_annotations = []
    for face_key, face_area in faces.items():
        x1, y1, x2, y2 = max(0, face_area[0]), max(0, face_area[1]), min(width, face_area[2]), min(height, face_area[3])
        annotation = [index_frame, face_key, x1, y1, x2 - x1, y2 - y1, 1, -1, -1, -1]  # Initial annotations
        initial_annotations.append(annotation)

    return frame_rgb, initial_annotations

def predict_age_and_gender(frame_rgb, initial_annotations, model_a_g, image_size):
    """
    Predicts age and gender for detected faces using the given model.

    Parameters:
    frame_rgb (np.ndarray): The RGB image array.
    initial_annotations (list): Initial annotations from face detection.
    model_a_g: The model used for age and gender estimation.
    image_size (int): The size to which the face image should be resized.

    Returns:
    List: A list of annotations including age and gender.
    """
    transform = get_val_transform(image_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extended_annotations = []

    for annotation in initial_annotations:
        _, face_key, x1, y1, width, height, _, _, _, _ = annotation
        x2, y2 = x1 + width, y1 + height

        # Extract the face
        face = frame_rgb[y1:y2, x1:x2]

        # Convert the face to PIL image and apply transformation
        face_pil = Image.fromarray(face)
        input_image = transform(face_pil).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            age_output, gender_output = model_a_g(input_image)
            predicted_age_group = nn.Softmax(dim=1)(age_output)
            age_group = predicted_age_group.argmax(dim=1).item()
            gender = gender_output.argmax(dim=1).item()

        # Extend the annotation
        extended_annotation = annotation + [age_group, gender]
        extended_annotations.append(extended_annotation)

    return extended_annotations

def annotate_image_with_predictions(frame_rgb, extended_annotations):
    """
    Annotates the image with the predictions for age and gender.

    Parameters:
    frame_rgb (np.ndarray): The RGB image array.
    extended_annotations (list): Annotations including age and gender predictions.

    Returns:
    np.ndarray: The annotated image.
    """
    for annotation in extended_annotations:
        _, _, x1, y1, width, height, _, _, _, _, age_group, gender = annotation
        x2, y2 = x1 + width, y1 + height

        # Annotate the image using OpenCV
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_rgb, str(age_group), (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame_rgb, str(gender), (x1, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame_rgb



#write Dataset, could be simpler
class VideoDataset(Dataset):
    def __init__(self, video_path, view="none"):
        """
        Args: video_path (str): Path to the video file
        """
        # Initialize the VideoReader
        self.vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # Load video in CPU memory
        self.view= view
        if view in ['top']: #'coming_in','going_out'
            self.x_th_frame = 100
        else:
            self.x_th_frame = 1
        self.length = int(len(self.vr)/self.x_th_frame)  # Total number of frames
        self.video_size = self.get_view(self.vr[0].asnumpy(), view = self.view).shape
        
        print(f"Video len: {self.length}")
    
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

# Process video and save annotated video
def process_video(video_path, model_a_g, model_face_detection, output_annotations_path, output_video_path = None, image_size=448):
    """
    ouptut_video_path: path to save the annotated video or None if no video should be saved
    """
    dataset = VideoDataset(video_path)

    video_writer = None
    annotations = []

    for idx, frame in enumerate(tqdm(range(len(dataset)))):
        frame = dataset[idx]
        
        frame_rgb, initial_annotations = detect_faces_and_annotate_initial(frame_rgb=frame, model_face_detection=model_face_detection, index_frame=idx)
        extended_annotations = predict_age_and_gender(frame_rgb, initial_annotations, model_a_g=model_a_g, image_size=image_size)
        annotated_frame = annotate_image_with_predictions(frame_rgb, extended_annotations)


        # Initialize video writer
        if output_video_path is not None:
            if video_writer is None:
                height, width, _ = annotated_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

            video_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        for annotation in extended_annotations:
            annotation[0] = idx  # Set frame index
            annotations.append(annotation)

    if output_video_path is not None:
        video_writer.release()

    # Save annotations in MOT style
    with open(output_annotations_path, 'w') as f:
        for annotation in annotations:
            f.write(','.join(map(str, annotation)) + '\n')

    print(f"Annotated video saved to {output_video_path}")
    print(f"Annotations saved to {output_annotations_path}")