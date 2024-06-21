import pandas as pd
import os
import decord as de
import cv2
from tqdm import tqdm, trange
from collections import Counter


def process_eyetracking_data(eye_tracking_data_path, video_path):
    """
    creates a dataframe with the eye tracking data and the corresponding frame number with 
    the x and y position of the gaze point adjusted to the video resolution
    """
    video = de.VideoReader(video_path)
    data = pd.read_csv(eye_tracking_data_path)
    w, h = video[0].shape[1], video[0].shape[0]
    data["pos_x"] = data["norm_pos_x"] * w
    data["pos_y"] = (1 - data["norm_pos_y"]) * h
    data = data.drop(columns=['topic'])
    return data


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    # Compute the area of the intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    # Compute the area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    # Compute the area of the union
    union_area = box1_area + box2_area - inter_area
    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def calculate_tracklets(df, iou_threshold=0.5):
    active_groups_curr_frame = []
    active_groups_last_frame = []
    retired_groups = []
    frame_old = -1

    for i in trange(len(df)):
        current_box = (df['x_l'].iloc[i], df['y_l'].iloc[i], df['width'].iloc[i], df['height'].iloc[i])
        frame = df['frame'].iloc[i]
        if frame != frame_old:
            retired_groups.extend(active_groups_last_frame)
            active_groups_last_frame = active_groups_curr_frame
            active_groups_curr_frame = []
        frame_old = frame 
        matched_groups = False

        for group_idx, group in enumerate(active_groups_last_frame):
            # Get the last detection of the group to compare
            last_idx = group[-1]
            last_box = (df['x_l'].iloc[last_idx], df['y_l'].iloc[last_idx], df['width'].iloc[last_idx], df['height'].iloc[last_idx])
            last_frame = df['frame'].iloc[last_idx]

            if frame == last_frame + 1 and calculate_iou(last_box, current_box) > iou_threshold:
                active_groups_curr_frame.append(active_groups_last_frame.pop(group_idx) + [i])
                matched_groups = True
                break

        if not matched_groups:
            # Start a new group if no existing group matches
            active_groups_curr_frame.append([i])

    # Finalize all groups with majority vote
    all_groups = active_groups_last_frame + active_groups_curr_frame + retired_groups
    return all_groups

def assign_majority_vote_with_iou(df):
    # Sort the dataframe by frame to ensure chronological order
    df = df.sort_values(by=['frame']).reset_index(drop=True)
    # Tracking multiple groups
    all_groups = calculate_tracklets(df, iou_threshold=0.5)

    # Initialize the new columns for majority vote assignment
    df['majority_gender'] = df['gender']
    df['majority_age'] = df['age_class']
    for group in tqdm(all_groups):
        genders = [df['gender'].iloc[idx] for idx in group]
        ages = [df['age_class'].iloc[idx] for idx in group]
        majority_gender = Counter(genders).most_common(1)[0][0]
        majority_age = Counter(ages).most_common(1)[0][0]
        
        for idx in group:
            df.at[idx, 'majority_gender'] = majority_gender
            df.at[idx, 'majority_age'] = majority_age
    return df

def enlarge_bounding_box(box, scale_factor:float=1.5, small_dim:bool=True):
    x1, y1, x2, y2 = box
    # Calculate the center of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the new half-width and half-height
    half_width = (x2 - x1) / 2
    half_height = (y2 - y1) / 2
        
    if half_height > half_width:
        if small_dim:
            enlarged_length = half_width * (scale_factor - 1)
        else:
            enlarged_length = half_height * (scale_factor - 1)  
    else:
        if small_dim:
            enlarged_length = half_height * (scale_factor - 1)
        else:
            enlarged_length = half_width * (scale_factor - 1)  
    new_half_width = half_width + enlarged_length
    new_half_height = half_height + enlarged_length

    # Calculate the new coordinates of the bounding box
    new_x1 = center_x - new_half_width
    new_y1 = center_y - new_half_height
    new_x2 = center_x + new_half_width
    new_y2 = center_y + new_half_height

    return new_x1, new_y1, new_x2, new_y2

def check_gaze_in_boxes(data_per_frame, scale_factor=1.5):
    """
    Subsets the box annotations to where the gaze point is within the box and returns the bounding box and the age class enriched with the gaze point.
    Parameters:
    - data_per_frame: pd.DataFrame
        The processed eye tracking data with gaze points and bounding boxes.
    - scale_factor: float
        The factor by which to enlarge the bounding boxes.
    """
    # Initialize the "eye_in_box" column to 0
    data_per_frame["eye_in_box"] = 0
    print("Checking gaze points in bounding boxes")
    for idx in trange(len(data_per_frame)):
        pos_x = data_per_frame.loc[idx, 'pos_x']
        pos_y = data_per_frame.loc[idx, 'pos_y']
        
        # Skip rows with NaN values for gaze points
        if pd.isna(pos_x) or pd.isna(pos_y):
            continue
        
        # Check if the gaze point is within any face box
        frame_number = data_per_frame.loc[idx, "frame"]
        frame_annotations = data_per_frame[data_per_frame["frame"] == frame_number]
        
        for ann_idx, row in frame_annotations.iterrows():
            if pd.isna(row["x_l"]):
                continue
            x_l, y_l, width, height = row["x_l"], row["y_l"], row["width"], row["height"]
            # Enlarge bounding box
            enlarged_box = enlarge_bounding_box((x_l, y_l, x_l + width, y_l + height), scale_factor)
            new_x1, new_y1, new_x2, new_y2 = map(int, enlarged_box)

            # Check if the gaze point is inside the enlarged bounding box
            if new_x1 <= pos_x <= new_x2 and new_y1 <= pos_y <= new_y2:
                # Update the eye_in_box for this specific bounding box
                data_per_frame.loc[ann_idx, "eye_in_box"] = 1

    # Return the updated DataFrame
    return data_per_frame

def annotate_video_eye_and_box(video_path, eyes_and_box_df, output_video_path):
    # Initialize decord video reader
    video_reader = de.VideoReader(video_path)

    # Prepare to write the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_reader.get_avg_fps()
    width, height = video_reader[0].shape[1], video_reader[0].shape[0]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = len(video_reader)
    print(f"Annotating {os.path.basename(video_path)} ")

    for idx in tqdm(range(frame_count)):
        frame = video_reader[idx].asnumpy()
        frame[:, :, [0, 2]] = frame[:, :, [2, 0]] # Convert RGB to BGR

        # Annotate the frame with the scatter point
        face_boxes = eyes_and_box_df[eyes_and_box_df['frame'] == idx]
        for _, row in face_boxes.iterrows():
            if row['eye_in_box'] == 1:
                use_color = (0, 255, 0)
            else:
                use_color = (0, 0, 255)
            if not pd.isna(row['pos_x']):
                pos_x, pos_y = int(row['pos_x']), int(row['pos_y'])
                cv2.circle(frame, (pos_x, pos_y), radius=5, color=use_color, thickness=-1)
            if pd.isna(row['x_l']):
                continue
            x_l, y_l, width, height = int(row['x_l']), int(row['y_l']), int(row['width']), int(row['height'])

            cv2.rectangle(frame, (x_l, y_l), (x_l + width, y_l + height), use_color, 2)
            cv2.putText(frame, f'Age: {row["age_class"]}', (x_l, y_l + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            gender = "m" if row["gender"]==1 else "f"
            cv2.putText(frame, f'Gender: {gender}', (x_l, y_l + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # Annotate the frame number
        cv2.putText(frame, f'Frame: {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2, cv2.LINE_AA)

        out.write(frame)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
