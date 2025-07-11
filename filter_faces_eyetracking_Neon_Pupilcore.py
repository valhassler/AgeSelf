# Use annotations of Bounding boxe and eye tracking data to check where the focus of the view is and to produce a video with the bounding boxes and the gaze point
# It also uses a simple tracking form where the labels are tracked over the frames and the majority vote is used to assign the label to the box

import pandas as pd
import os
from ageself.filter_faces_eyetracking_functions import process_eyetracking_data, assign_majority_vote_with_iou, check_gaze_in_boxes, annotate_video_eye_and_box, calculate_tracklets, assign_large_group
from ageself.filter_faces_eyetracking_functions import smooth_running_median, build_eye_tracking_dataset,build_min_expansion_df,make_colour_map

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


base_path = "/usr/users/vhassle"

# Input paths
base_eye_tracking_raw_path = os.path.join(base_path,"datasets/Wortschatzinsel/head_mounted_data/eye_tracking_annotations/valid") #Where the original eye tracking data is stored and also the original videos kinda
box_annotation_base_path = os.path.join(base_path,"model_outputs/Wortschatzinsel/age_gender_classification_reversed_from_results") #Where Face boxes and there age annotation is stored

# Output paths
output_folder_path = os.path.join(base_path,"model_outputs/Wortschatzinsel/detection_tracking_merged_v04", os.path.basename(box_annotation_base_path)+"")
os.makedirs(output_folder_path, exist_ok=True)


# Information processing
information_raw_df = pd.read_csv(os.path.join(base_path,"datasets/Wortschatzinsel/head_mounted_data/scene_view_creation_df.csv"))
decision_df = pd.read_csv(os.path.join(base_path,"datasets/Wortschatzinsel/head_mounted_data/final_IDs.csv"))

# Create one dataset that has the annotations from the facedetection boxes and their according labels 
information_filtered_df = pd.merge(information_raw_df, decision_df, left_on="scene_view_nr", right_on="ID", how="inner")
print(set(decision_df["ID"]) - set(information_raw_df["scene_view_nr"]))
empty_eye_tracking = []

generation = ""

information_filtered_df = information_filtered_df.sort_values(by=["scene_view_nr"])

for index, row in information_filtered_df.iterrows():
    seq_name = row["new_name"]

    # video input (just necessary if video is created )
    video_path = os.path.join(base_eye_tracking_raw_path,"../../videos/valid", f"{seq_name}.mp4")

    #annoted boxes input
    box_annotation_path = os.path.join(box_annotation_base_path, f"{seq_name}{generation}.txt")

    #output paths
    output_csv_path = os.path.join(output_folder_path, f"{seq_name}_annotated.csv")
    output_video_path = os.path.join(output_folder_path, f"{seq_name}_annotated_better.mp4")

    eye_tracking_raw_path = os.path.join(base_eye_tracking_raw_path, str(seq_name), "gaze_positions.csv")
    is_neon = row["neon"]
    data_per_frame_eye, gaze_df = build_eye_tracking_dataset(seq_name, video_path, eye_tracking_raw_path, base_eye_tracking_raw_path, is_neon)
    
    box_annotation_df = pd.read_csv(box_annotation_path, header=None)
    box_annotation_df.columns = ["frame", "face_nuber_on_frame", "x_l", "y_l", "width", "height", "n1","n2","n3","n4","age_class", "gender"]
    box_annotation_df.frame = box_annotation_df.frame # change stuf if necessary
    box_eye_annotation_df = pd.merge(box_annotation_df, data_per_frame_eye, how='outer', left_on='frame', right_on='world_index')

    box_eye_annotation_df['frame'] = box_eye_annotation_df['frame'].combine_first(box_eye_annotation_df['world_index'])
    box_eye_annotation_df.drop(columns=['world_index'], inplace=True)

    #appends a column to the box_eye_annotation_df where it is checked if the gaze point is within the bounding box
    box_eye_annotation_df = box_eye_annotation_df.sort_values(by=['frame']).reset_index(drop=True)
    all_groups = calculate_tracklets(box_eye_annotation_df, iou_threshold=0.5)
    box_eye_annotation_df = assign_majority_vote_with_iou(box_eye_annotation_df, all_groups)
    box_eye_annotation_df["large_group"] = assign_large_group(box_eye_annotation_df, all_groups, group_size_threshold = 3)

    # add in count of grown up child per frame
    box_eye_annotation_df["n_adults_headmounted"] = box_eye_annotation_df["age_class"] == 2
    box_eye_annotation_df["n_children_headmounted"] = box_eye_annotation_df["age_class"] < 2
    box_eye_annotation_grouped = box_eye_annotation_df.groupby("frame").agg("sum")
    box_eye_annotation_df.drop(columns=["n_children_headmounted","n_adults_headmounted"], inplace=True)
    box_eye_annotation_grouped["frame"] = box_eye_annotation_grouped.index
    box_eye_annotation_grouped.index.name = None
    box_eye_annotation_df = box_eye_annotation_df.merge(box_eye_annotation_grouped[["frame","n_adults_headmounted","n_children_headmounted"]], on="frame", how="left")

    #which sizes of bounding box expansion to use
    expansion_pixels = [0,5,10,15,20,30,40,50,60,80,100]
    expansion_pixels = [int(pixel) if row["neon"] else pixel for pixel in expansion_pixels]

    for expansion_pixel in expansion_pixels:
        scale_factor = 1.0
        tracked_box_eye_annotation_df = check_gaze_in_boxes(box_eye_annotation_df, scale_factor=scale_factor, expansion_pixels=expansion_pixel)
        # if in the box is not part of a large group it is ignored for :    data_per_frame.loc[ann_idx, "eye_in_box"]         box_eye_annotation_df = box_eye_annotation_df[box_eye_annotation_df["large_group"] == True]
        tracked_box_eye_annotation_df["eye_in_box"] = tracked_box_eye_annotation_df["eye_in_box"] * tracked_box_eye_annotation_df["large_group"]


        base_name = os.path.basename(box_annotation_path).split(".")[0]
        output_csv_path = os.path.join(output_folder_path, f"{seq_name}_scale_{scale_factor}_expansion_{expansion_pixel:03d}.csv")
        #convert number to date
        if not row["neon"]:
            ts = pd.to_datetime(tracked_box_eye_annotation_df["timestamp"],unit="ns",errors="coerce")
        else:
            ts = pd.to_datetime(tracked_box_eye_annotation_df["timestamp"],unit="ms",errors="coerce")

        tracked_box_eye_annotation_df["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        tracked_box_eye_annotation_df.to_csv(output_csv_path, index=False)

    min_exp_df = build_min_expansion_df(box_eye_annotation_df, expansion_pixels[1::3])

    # 2. build a colour map (one distinct colour per expansion size)
    colour_map = make_colour_map(expansion_pixels[1::3])
    # 3. annotate the video
    print("Annotating video with eye tracking and bounding boxes...")
    annotate_video_eye_and_box(video_path,
                            min_exp_df,
                            output_video_path,
                            colour_map)

    print(empty_eye_tracking)