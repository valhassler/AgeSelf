import pandas as pd
import os
import decord as de
import cv2
from tqdm import tqdm, trange
from collections import Counter
import numpy as np

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
    print("Calculating tracklets")
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

def assign_majority_vote_with_iou(df: pd.DataFrame, all_groups: list):
    """
    ### assign_majority_vote_with_iou
    Assign majority gender and age class to each group based on Intersection over Union (IoU) results.

    **Args:**
    - `df` (pd.DataFrame): Input DataFrame containing the columns `gender` and `age_class`.
    - `all_groups` (list): List of lists, where each inner list contains the indices of a group.

    **Returns:**
    - `pd.DataFrame`: Updated DataFrame with majority gender and age class assigned to each group.

    **Behavior:**
    - For each group of indices, the majority `gender` and `age_class` are calculated.
    - The majority values are assigned to all entries in the group.
    - Original columns `gender` and `age_class` are updated in place.
    """
    print("Assigning majority vote with IoU...")

    # Iterate over each group and calculate majority vote
    for group in tqdm(all_groups, desc="Processing groups"):
        # Extract genders and ages for the group
        group_genders = df.loc[group, 'gender']
        group_ages = df.loc[group, 'age_class']

        # Compute majority vote for gender and age_class
        majority_gender = Counter(group_genders).most_common(1)[0][0]
        majority_age = Counter(group_ages).most_common(1)[0][0]

        # Assign majority values back to the group
        df.loc[group, 'gender'] = majority_gender
        df.loc[group, 'age_class'] = majority_age

    return df


def assign_large_group(df: pd.DataFrame, all_groups: list, group_size_threshold: int = 3):
    """
    ### assign_large_group
    Assigns a boolean flag to indicate whether a group is "large" based on its size.

    **Args:**
    - `df` (pd.DataFrame): Input DataFrame where the `large_group` column will be added or updated.
    - `all_groups` (list): List of lists, where each inner list contains the indices of a group.
    - `group_size_threshold` (int, optional): Minimum size to consider a group as "large". Defaults to 3.

    **Returns:**
    - `pd.DataFrame`: Updated DataFrame with the `large_group` column indicating group size status.
    """
    print("Assigning large group status...")
    
    # Initialize the 'large_group' column as False
    df['large_group'] = False
    
    # Process each group
    for group in tqdm(all_groups, desc="Processing groups"):
        if len(group) >= group_size_threshold:
            # Efficiently set 'large_group' for all indices in the current group
            df.loc[group, 'large_group'] = True
    
    return df['large_group']

def enlarge_bounding_box(
    box,
    scale_factor: float = 1.5,
    small_dim: bool = True,
    expansion_pixels: int = 0
):
    """
    Enlarge a bounding box by first scaling around its center, then adding a fixed pixel margin.

    Args:
        box (tuple): (x1, y1, x2, y2)
        scale_factor (float): Multiplicative scale (e.g. 1.5).
        small_dim (bool): If True, scale the smaller dimension by scale_factor; otherwise, scale the larger.
        expansion_pixels (float): Additional pixels to expand on each side after scaling.

    Returns:
        new_x1, new_y1, new_x2, new_y2
    """
    x1, y1, x2, y2 = box
    # Center
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # Original half-dimensions
    half_width = (x2 - x1) / 2.0
    half_height = (y2 - y1) / 2.0

    # Determine which half-dimension to scale
    if (half_height > half_width and not small_dim) or (half_height <= half_width and small_dim):
        new_half_width = half_height * scale_factor
        new_half_height = half_height * scale_factor
    else:
        new_half_width = half_width * scale_factor
        new_half_height = half_width * scale_factor

    # Add fixed pixel expansion on all sides
    new_half_width += expansion_pixels
    new_half_height += expansion_pixels

    # Compute new coordinates
    new_x1 = center_x - new_half_width
    new_y1 = center_y - new_half_height
    new_x2 = center_x + new_half_width
    new_y2 = center_y + new_half_height

    return new_x1, new_y1, new_x2, new_y2


def check_gaze_in_boxes(data_per_frame, scale_factor=1.5, expansion_pixels=0):
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
        pos_x = data_per_frame.iloc[idx]['pos_x']
        pos_y = data_per_frame.iloc[idx]['pos_y']
        
        # Skip rows with NaN values for gaze points
        if pd.isna(pos_x) or pd.isna(pos_y):
            continue
        
        # Check if the gaze point is within any face box
        frame_number = data_per_frame.iloc[idx]["frame"]
        frame_annotations = data_per_frame[data_per_frame["frame"] == frame_number]
        
        for ann_idx, row in frame_annotations.iterrows():
            if pd.isna(row["x_l"]):
                continue
            x_l, y_l, width, height = row["x_l"], row["y_l"], row["width"], row["height"]
            # Enlarge bounding box
            enlarged_box = enlarge_bounding_box((x_l, y_l, x_l + width, y_l + height), scale_factor, expansion_pixels=expansion_pixels)
            new_x1, new_y1, new_x2, new_y2 = map(int, enlarged_box)

            # Check if the gaze point is inside the enlarged bounding box
            if new_x1 <= pos_x <= new_x2 and new_y1 <= pos_y <= new_y2:
                # Update the eye_in_box for this specific bounding box
                data_per_frame.loc[ann_idx, "eye_in_box"] = 1

    # Return the updated DataFrame
    return data_per_frame


def build_min_expansion_df(
        box_eye_annotation_df: pd.DataFrame,
        expansion_pixels: list[int],
        scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Returns a copy of `box_eye_annotation_df` with one extra column
    (`min_expansion`) that stores the smallest expansion size whose
    box still contains the gaze **and** belongs to a large group.
    NaN  →  there is no box that fulfils the criteria.
    """
    df = box_eye_annotation_df.copy()
    df["min_expansion"] = np.nan

    for exp in sorted(expansion_pixels):          # small → large
        tmp = check_gaze_in_boxes(box_eye_annotation_df,
                                  scale_factor=scale_factor,
                                  expansion_pixels=exp)

        inside = (tmp["eye_in_box"] == 1) & (tmp["large_group"] == True)
        needs_update = inside & df["min_expansion"].isna()
        df.loc[needs_update, "min_expansion"] = exp

    return df

import colorsys

def make_colour_map(expansion_pixels: list[int]) -> dict[int, tuple[int, int, int]]:
    """
    Evenly spreads hues over the number of expansion sizes and
    converts them to BGR ints (0–255) for OpenCV.
    """
    n = len(expansion_pixels)
    colour_map = {}

    for i, exp in enumerate(sorted(expansion_pixels)):
        h = i / n                       # 0 … <1
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        colour_map[exp] = (int(b*255), int(g*255), int(r*255))  # BGR
    return colour_map


def annotate_video_eye_and_box(
        video_path: str,
        df: pd.DataFrame,                   # ← contains the new 'min_expansion' column
        output_video_path: str,
        colour_map: dict[int, tuple[int, int, int]],
):
    import os
    import cv2
    from tqdm import tqdm
    import decord as de
    import pandas as pd

    video_reader = de.VideoReader(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_reader.get_avg_fps()
    H, W = video_reader[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    print(f"Annotating {os.path.basename(video_path)} …")
    for idx in tqdm(range(len(video_reader))):
        frame = video_reader[idx].asnumpy()
        frame[:, :, [0, 2]] = frame[:, :, [2, 0]]        # RGB → BGR

        rows = df[df["frame"] == idx]

        # ──── 1. draw boxes (original + yellow expansions) ────
        for _, row in rows.iterrows():
            if pd.isna(row["x_l"]) or not bool(row["large_group"]):
                continue

            x, y = int(row["x_l"]), int(row["y_l"])
            w, h = int(row["width"]), int(row["height"])

            # original box (thicker, green/red)
            base_colour = (0, 255, 0) if row["eye_in_box"] == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), base_colour, 2)

            # expansion boxes (yellow, thin)
            for extra in colour_map:                       # use the keys, not the values
                if extra == 0:
                    continue
                x1 = max(0, x - extra)
                y1 = max(0, y - extra)
                x2 = min(W-1, x + w + extra)
                y2 = min(H-1, y + h + extra)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # optional text (age, gender)
            cv2.putText(frame, f'Age: {row["age_class"]}', (x, y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            gender = "m" if row["gender"] == 1 else "f"
            cv2.putText(frame, f'Gender: {gender}', (x, y+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # ──── 2. draw the pointer once per frame ────
        #    • first valid (pos_x,pos_y) sets location
        #    • smallest expansion with gaze‑inside sets colour
        pointer_coord, ptr_colour = None, None
        min_exp_this_frame = rows["min_expansion"].dropna().min()
        if not np.isnan(min_exp_this_frame):
            ptr_colour = colour_map[int(min_exp_this_frame)]

        # take the first non‑nan gaze point for the frame (if any)
        for _, row in rows.iterrows():
            if not pd.isna(row["pos_x"]) and not pd.isna(row["pos_y"]):
                pointer_coord = (int(row["pos_x"]), int(row["pos_y"]))
                break

        if pointer_coord is not None:
            core_colour = ptr_colour if ptr_colour is not None else (80, 80, 80)  # grey if never inside
            inner_r = 5
            cv2.circle(frame, pointer_coord, inner_r, core_colour, -1)      # coloured core
            cv2.circle(frame, pointer_coord, inner_r+2, (0, 255, 0), 2)     # green rim

        # ──── 3. frame counter ────
        cv2.putText(frame, f'Frame: {idx}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def load_world_timestamps(path):
    """
    Load timestamps from a .csv or .npy file, coerce to int, and sort.
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        df = df.rename(columns={'timestamp [ns]': 'timestamp'})

    elif path.endswith('.npy'):
        arr = np.load(path)
        df = pd.DataFrame(arr, columns=['timestamp'])

    else:
        raise ValueError(f"Unsupported extension: {os.path.splitext(path)[1]}")

    df.sort_values(by='timestamp', ignore_index=True, inplace=True)
    return df


def load_and_preprocess_gaze(path, W, H, min_conf=0.6, has_confidence=True):
    """
    Load raw gaze CSV, optionally filter by confidence, convert normalized coords to pixels,
    drop out-of-bounds, and sort by timestamp.
    """
    df = pd.read_csv(path)
    if has_confidence:
        df = df[df['confidence'] >= min_conf]

    df = df.rename(columns={
        'gaze_timestamp': 'timestamp',
        'norm_pos_x': 'pos_x',
        'norm_pos_y': 'pos_y'
    })[['timestamp', 'pos_x', 'pos_y']]

    df['pos_x'] = df['pos_x'] * W
    df['pos_y'] = (1 - df['pos_y']) * H

    # Mask out-of-bounds
    df.loc[~df['pos_x'].between(-0.1*W, 1.1*W), 'pos_x'] = np.nan
    df.loc[~df['pos_y'].between(-0.1*H, 1.1*H), 'pos_y'] = np.nan

    df.loc[df['pos_x'].isna(), 'pos_y'] = np.nan
    df.loc[df['pos_y'].isna(), 'pos_x'] = np.nan

    df.sort_values('timestamp', ignore_index=True, inplace=True)
    return df


def build_eye_tracking_dataset(seq_name:str, video_path:str, eye_tracking_raw_path:str, base_eye_tracking_raw_path:str, is_neon:bool, window_ms = 100) -> pd.DataFrame:
    # 1. Determine timestamp file path
    if is_neon:
        ts_path = os.path.join(base_eye_tracking_raw_path, seq_name, 'world_timestamps.csv')
    else:
        ts_path = os.path.join(
            os.path.dirname(video_path),
            '..', 'world_ts_pupilcore',
            seq_name.split('_')[0] + '.npy'
        )

    # 2. Load & sort world timestamps
    world_ts = load_world_timestamps(ts_path)

    # 3. Load & preprocess gaze data
    if is_neon:
        gaze_df = pd.read_csv(eye_tracking_raw_path)
        gaze_df.columns = ['timestamp', 'pos_x', 'pos_y']
        # gaze_df = load_and_preprocess_gaze(
        #     eye_tracking_raw_path,
        #     W=1600, H=1200,
        #     min_conf=0.0,
        #     has_confidence=False
        # )
    else:
        gaze_df = load_and_preprocess_gaze(
            eye_tracking_raw_path,
            W=1024, H=768,
            min_conf=0.6,
            has_confidence=True
        )

    # 4. Merge with nearest timestamp
    if is_neon:
        world_ts['timestamp'] = world_ts['timestamp'].astype(int)
        gaze_df['timestamp'] = gaze_df['timestamp'].astype(int)
    data = pd.merge_asof(world_ts, gaze_df, on='timestamp', direction='nearest')

    # 5. Add world_index
    data = data.reset_index(drop=True).rename_axis('world_index').reset_index()

    # 6. Post-process non-neon timestamps and smoothing
    if not is_neon:
        data['timestamp'] = data['timestamp'] * 1000
        if window_ms > 0:
            data = smooth_running_median(data, window_ms=window_ms, min_periods=1)

    return data, gaze_df

def smooth_running_median(data_frame, window_ms=100.0, min_periods=1):
    """
    Applies a running median to the 'pos_x' and 'pos_y' columns of a DataFrame.
    Args:
        data_frame (pd.DataFrame): DataFrame with 'pos_x', 'pos_y', and 'timestamp' columns.
        window_ms (float): Window size in milliseconds for the rolling median.
        min_periods (int): Minimum number of observations in the window required to have a value.
    Returns:
        pd.DataFrame: A copy of the input DataFrame with 'pos_x' and 'pos_y' smoothed.
    """
    data_frame = data_frame.copy()
    td_index = pd.to_timedelta(data_frame["timestamp"], unit="ms")
    data_frame = data_frame.set_index(td_index)
    win = f"{window_ms}ms"
    roll = data_frame[["pos_x", "pos_y"]].rolling(win, center=True, min_periods=min_periods).median()
    data_frame.loc[:, ["pos_x", "pos_y"]] = roll
    data_frame = data_frame.reset_index(drop=True)
    return data_frame
