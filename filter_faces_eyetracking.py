import pandas as pd
import os
from ageself.filter_faces_eyetracking_functions import process_eyetracking_data, assign_majority_vote_with_iou, check_gaze_in_boxes, annotate_video_eye_and_box
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


video_name = "raspi_wsi5_2024_05_19_001.mp4"
eye_tracking_data_name = "2024_05_19_001_cal_SI-fd3b8912-f5c3.csv"

base_path_wd = "/usr/users/vhassle/psych_track/integrate_eyetracking"
base_path_eyetracking_data = os.path.join(base_path_wd, "eyetracking_data")
eye_tracking_data_path = os.path.join(base_path_eyetracking_data, eye_tracking_data_name)

video_base_path = "/usr/users/vhassle/datasets/Wortschatzinsel/all_videos"
video_name_short = video_name.split(".")[0]
video_path = os.path.join(video_base_path, video_name)

box_annotation_base_path = "/usr/users/vhassle/model_outputs/outputs_AgeSelf/" #Where Face boxes and there age annotation is stored
model_name_classification = "faces_a_g_img_size_150_rot_90"
generation = "r003"
box_annotation_path = os.path.join(box_annotation_base_path, model_name_classification, f"{video_name_short}_{generation}.txt")

base_output_path = os.path.join(base_path_wd, "output", model_name_classification)
os.makedirs(base_output_path, exist_ok=True)

data = process_eyetracking_data(eye_tracking_data_path, video_path)

#melt it down to one annotation per frame
data_per_frame_eye = data.groupby('world_index').median().reset_index()

box_annotation_df = pd.read_csv(box_annotation_path, header=None)
box_annotation_df.columns = ["frame", "face_nuber_on_frame", "x_l", "y_l", "width", "height", "n1","n2","n3","n4","age_class", "gender"]


box_eye_annotation_df = pd.merge(box_annotation_df, data_per_frame_eye, how='outer', left_on='frame', right_on='world_index')
box_eye_annotation_df['frame'] = box_eye_annotation_df['frame'].combine_first(box_eye_annotation_df['world_index'])
box_eye_annotation_df.drop(columns=['world_index'], inplace=True)

# box_annotation_path = "/usr/users/vhassle/model_outputs/outputs_AgeSelf/age_classification_model_20_focal/raspi_wsi5_2024_05_19_001_r002.txt"
scale_factor = 1.5
#appends a column to the box_eye_annotation_df where it is checked if the gaze point is within the bounding box
box_eye_annotation_df = assign_majority_vote_with_iou(box_eye_annotation_df)
box_eye_annotation_df["age_class"] = box_eye_annotation_df["majority_age"]
box_eye_annotation_df["gender"] = box_eye_annotation_df["majority_gender"]

tracked_box_eye_annotation_df = check_gaze_in_boxes(box_eye_annotation_df, scale_factor=scale_factor)

output_csv_path = os.path.join(base_output_path, f"{video_name_short}_annotated.csv")
output_video_path = os.path.join(base_output_path, f"{video_name_short}_annotated_better.mp4")

tracked_box_eye_annotation_df.to_csv(output_csv_path, index=False)

annotate_video_eye_and_box(video_path, tracked_box_eye_annotation_df, output_video_path)