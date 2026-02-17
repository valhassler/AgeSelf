"""
hmet_frames_to_global_times.py
2025-06-02
Sebastian Isbaner, sebastian.isbaner@uni-goettingen.de
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tqdm
import os

BASE_PATH = Path(
    "/run/user/715136/gvfs/"
    "smb-share:server=wfs-biologie.top.gwdg.de,share=ubps-all$/"
    "Language_Acquisition/HIWI/HIWIstudies/OLDstudiesHIWI/"
    "SmallFaces in the Wild HIWI/Data"
)

HMET_DATA_FOLDER = BASE_PATH / "Neon"
top_view_tracking_results_path = BASE_PATH /"Top_view_tracking_results"/ "merged_top_view_tracking_results.csv"
ENV_DATA = pd.read_csv(top_view_tracking_results_path, parse_dates=['frame_time'])
ENV_DATA.rename(columns={'adults': 'n_adults', 'children': 'n_children'}, inplace=True)

def load_timestamps(id, hmet_data_folder):
    """
    Load timestamps for a given ID from the hmet data folder.
    """
    
    if int(id) < 300 or int(id) >= 500:
        if (id == '505') or (id == '622'):
            timestamps_file = list(HMET_DATA_FOLDER.parent.glob(f'Pupil Core/**/{id}*/world_timestamps.npy'))
        else:
            timestamps_file = list(HMET_DATA_FOLDER.parent.glob(f'Pupil Core/**/{id}*/**/exports/*/world_timestamps.npy'))
        if not timestamps_file:
            raise FileNotFoundError(f'No timestamp file found for {id}.')
        if len(timestamps_file) > 1:
            if timestamps_file[0].parent.parent == timestamps_file[1].parent.parent:
                # if both files are in the same parent folder, they are from the same recording
                timestamps_file = timestamps_file[-1] # take the most recent one
            else:
                raise FileNotFoundError(f'Multiple timestamp files found for {id}.')
        else: 
            timestamps_file = timestamps_file[0]
        timestamps = np.load(timestamps_file)
        # convert timestamps to unix datetime
        if (id == '505') or (id == '622'):
            info_data_file = timestamps_file.parent / 'info.player.json'
        else:
            info_data_file = timestamps_file.parent.parent.parent / 'info.player.json'
        with open(info_data_file, 'r') as f:
            info_data = json.load(f)
        start_time_synced = info_data.get('start_time_synced_s')
        start_time_system = info_data.get('start_time_system_s')
        # if id == '622':# this is a special case where the raspi was restarted and did not update the time poperly before the recording started. Therefore, the start time from the participant sheet is used (- 2 hours).
        #     start_time_system = pd.Timestamp('2024-06-16 11:16').timestamp()
        
        timestamps = timestamps - start_time_synced + start_time_system
        timestamps = pd.to_datetime(timestamps, unit='s') + pd.Timedelta(hours=2)  # Adjust for timezone UTC+1 (+1 summer time)
    else:
        timestamps_file = list(HMET_DATA_FOLDER.glob(f'**/{id}/**/neon_player/world_timestamps_unix.npy'))
        if not timestamps_file:
            raise FileNotFoundError(f'No timestamp file found for {id}.')
        if len(timestamps_file) > 1:
            raise FileExistsError(f'Multiple timestamp files found for {id}.')
        
        timestamps_file = timestamps_file[0]
        timestamps = np.load(timestamps_file)
        timestamps = pd.to_datetime(timestamps) + pd.Timedelta(hours=2)  # Adjust for timezone UTC+1 (+1 summer time)
        
    timestamps = timestamps.to_frame(name='timestamp').reset_index(drop=True)
    return timestamps
def assing_nr_children_adult_from_top_view(df, id):
    try:
        timestamps = load_timestamps(id, HMET_DATA_FOLDER)
        df['frame'] = df['frame'].astype(int)
        
        # merge the timestamps with the hmet data
        df = df.merge(timestamps, left_on='frame', right_index=True, how='left')
        # merge the tracking results
        df = pd.merge_asof(df, ENV_DATA, left_on='timestamp', right_on='frame_time', direction='nearest', allow_exact_matches=True)
    except Exception as e:
        print(f"Error processing ID {id}: {e}")
        return None
    # check difference between timestamps and frame_time
    time_diff = (df['timestamp'] - df['frame_time']).dt.total_seconds().abs()
    thres = 5.1 # threshold for the time difference in seconds
    if time_diff.max() > thres: # top view frames analyzed are 3 or 10 seconds apart
        if time_diff.min() > 5e5: # this is because the top view video on 16.06. is not available --> do not display warning
            pass
        else:
            tqdm.tqdm.write(f'Time difference between timestamps and frame_time is too large for ID {id}. Max difference: {time_diff.max()} seconds. Check this file!')
        df.loc[time_diff>thres,'n_adults'] = np.nan
        df.loc[time_diff>thres,'n_children'] = np.nan
    
    return df


if __name__ == "__main__":

    hmet_folder      = (BASE_PATH/ "head_mounted_tracking_results"/ "results_data"/ "output_with_eyetracking")
    hmet_files = list(hmet_folder.glob('*_annotated.csv'))
    hmet_files.sort()

    df = pd.read_csv(hmet_files[0])
    id = os.path.basename(hmet_files[0]).split('_')[0]
    id = int(id)

    assing_nr_children_adult_from_top_view(df, id)

    

    
