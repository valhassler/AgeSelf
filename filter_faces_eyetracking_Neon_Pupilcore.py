# Use annotations of Bounding boxe and eye tracking data to check where the focus of the view is and to produce a video with the bounding boxes and the gaze point
# It also uses a simple tracking form where the labels are tracked over the frames and the majority vote is used to assign the label to the box



import pandas as pd
import os
from ageself.filter_faces_eyetracking_functions import process_eyetracking_data, assign_majority_vote_with_iou, check_gaze_in_boxes, annotate_video_eye_and_box, calculate_tracklets, assign_large_group
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

use_base_path = "/mnt/lustre-emmy-hdd/usr/u11216"
base_path_eyetracking_data = os.path.join(use_base_path,"datasets/Wortschatzinsel/eye_tracking/head_mounted_data/eye_tracking_annotations") #Where the original eye tracking data is stored and also the original videos kinda
box_annotation_base_path = os.path.join(use_base_path,"outputs/age_gender_combined_training_r02_25v2") #Where Face boxes and there age annotation is stored

information_raw_df = pd.read_csv(os.path.join(use_base_path,"datasets/Wortschatzinsel/eye_tracking/head_mounted_data/scene_view_creation_df.csv"))
decision_df = pd.read_csv(os.path.join(use_base_path,"datasets/Wortschatzinsel/eye_tracking/head_mounted_data/final_IDs.csv"))

# Create one dataset that has the annotations from the facedetection boxes and their according labels 
information_filtered_df = pd.merge(information_raw_df, decision_df, left_on="scene_view_nr", right_on="ID", how="inner")
print(set(decision_df["ID"]) - set(information_raw_df["scene_view_nr"]))
empty_eye_tracking = []

for index, row in information_filtered_df.iterrows():
    # video_name = row["scene_view_nr"]
    video_name = row["new_name"]
    video_path = os.path.join(base_path_eyetracking_data,"../videos", f"{video_name}.mp4")

    generation = "r02_25v2"
    box_annotation_path = os.path.join(box_annotation_base_path, f"{video_name}_{generation}.txt")

    base_output_path = os.path.join(box_annotation_base_path, "output_with_eyetracking")
    os.makedirs(base_output_path, exist_ok=True)
    output_csv_path = os.path.join(base_output_path, f"{video_name}_annotated.csv")
    output_video_path = os.path.join(base_output_path, f"{video_name}_annotated_better.mp4")
    check_processed_path = os.path.join(base_output_path, f"{video_name}_processed.txt")

    if os.path.exists(output_csv_path) or os.path.exists(output_video_path) or os.path.exists(check_processed_path):
        print(f"already processed {video_name}")
        continue

    with open(check_processed_path, "w") as f:
        f.write("processed")



    path_to_gaze = os.path.join(base_path_eyetracking_data, str(video_name), "gaze_positions.csv")

    if row["neon"]:
        path_to_timestamps = os.path.join(base_path_eyetracking_data, video_name, "world_timestamps.csv")

        world_ts = pd.read_csv(path_to_timestamps)
        world_ts.columns = ["world_index"]
        world_ts.sort_values(by="world_index", ignore_index=True, inplace=True)    

        gaze_position_df = pd.read_csv(path_to_gaze)
        gaze_position_df.columns = ["world_index", "pos_x", "pos_y"]
        world_ts["world_index"] = world_ts["world_index"].apply(lambda x: int(x))

        data = pd.merge_asof(world_ts,gaze_position_df ,left_on="world_index", right_on="world_index", direction="nearest")
        data["world_index"] = [i for i in range(len(data))]
    else:
        if not os.path.exists(path_to_gaze):
            print(f"no gaze data for {video_name}")
            empty_eye_tracking.append(video_name)
            continue
        gaze_position_df = pd.read_csv(path_to_gaze)
        W, H = 1024, 768 #resolution of the pupilcore neon is 1600x1200

        data = gaze_position_df[["world_index", "norm_pos_x", "norm_pos_y"]]
        data.columns = ["world_index", "pos_x", "pos_y"]
        data.loc[:, "pos_x"] = data["pos_x"] * W
        data.loc[:, "pos_y"] = (1 - data["pos_y"]) * H

    data_per_frame_eye = data.groupby('world_index').median().reset_index()

    box_annotation_df = pd.read_csv(box_annotation_path, header=None)
    box_annotation_df.columns = ["frame", "face_nuber_on_frame", "x_l", "y_l", "width", "height", "n1","n2","n3","n4","age_class", "gender"]


    box_eye_annotation_df = pd.merge(box_annotation_df, data_per_frame_eye, how='outer', left_on='frame', right_on='world_index')
    box_eye_annotation_df['frame'] = box_eye_annotation_df['frame'].combine_first(box_eye_annotation_df['world_index'])
    box_eye_annotation_df.drop(columns=['world_index'], inplace=True)

    scale_factor = 1.5
    #appends a column to the box_eye_annotation_df where it is checked if the gaze point is within the bounding box

    box_eye_annotation_df = box_eye_annotation_df.sort_values(by=['frame']).reset_index(drop=True)
    all_groups = calculate_tracklets(box_eye_annotation_df, iou_threshold=0.5)
    box_eye_annotation_df = assign_majority_vote_with_iou(box_eye_annotation_df, all_groups)
    box_eye_annotation_df = assign_large_group(box_eye_annotation_df, all_groups)

    tracked_box_eye_annotation_df = check_gaze_in_boxes(box_eye_annotation_df, scale_factor=scale_factor)



    tracked_box_eye_annotation_df.to_csv(output_csv_path, index=False)
    annotate_video_eye_and_box(video_path, tracked_box_eye_annotation_df, output_video_path)

print(empty_eye_tracking)