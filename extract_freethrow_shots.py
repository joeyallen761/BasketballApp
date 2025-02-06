import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path ="Project4BasketballHumanProject-dev1-2025-02-06/videos/freethrows_38secDLC_resnet101_Project4BasketballHumanProjectFeb6shuffle1_103000_filtered.csv"
df = pd.read_csv(file_path, header=[1, 2], index_col=0)  # Multi-index header
joints = [body_part for body_part in df.columns.get_level_values(0).unique()]

threshold = 0.8
#df = df[df.xs('likelihood',level=1, axis=1) > threshold]
frame_rate = 30

def weighted_moving_average(data, likelihood, window_size=15):
    """
    Applies a weighted moving average to a time series using likelihood scores as weights.
    
    Args:
        data (pd.Series): The time series data (e.g., x or y coordinates of a joint).
        likelihood (pd.Series): Corresponding likelihood scores.
        window_size (int): The number of surrounding points to consider (half window).
        
    Returns:
        pd.Series: Smoothed data.
    """
    smoothed = np.zeros(len(data))

    for i in range(len(data)):
        # Define the window bounds
        start = max(i - window_size, 0)
        end = min(i + window_size + 1, len(data))
        
        # Extract values and weights
        values = data[start:end].values
        weights = likelihood[start:end].values

        # Compute weighted average
        weighted_sum = np.sum(values * weights)
        weight_total = np.sum(weights)

        # Avoid division by zero
        smoothed[i] = weighted_sum / weight_total if weight_total > 0 else data[i]

    return pd.Series(smoothed, index=data.index)

def get_shot_peaks(df):
    wrist1y = weighted_moving_average(df['wrist1']['y'], df['wrist1']['likelihood'])
    wrist2y = weighted_moving_average(df['wrist2']['y'], df['wrist2']['likelihood'])
    foreheady = weighted_moving_average(df['forehead']['y'], df['wrist2']['likelihood'])

    wrist1_likelihood_rolling = df['wrist1']['likelihood'].rolling(30).mean()
    valid_frame_indices = df.index[wrist1_likelihood_rolling>.6]

    wrist1_forehead_distance = (foreheady-wrist1y)
    wrist2_forehead_distance = (foreheady-wrist2y)

    shot_indicator_frames = valid_frame_indices[((wrist1_forehead_distance[valid_frame_indices]>0)&(wrist2_forehead_distance[valid_frame_indices]>0))]
    split_indices = []
    for i in range(1, len(shot_indicator_frames)):
        curr_frame = shot_indicator_frames[i]
        prev_frame = shot_indicator_frames[i-1]
        if curr_frame - prev_frame > frame_rate: # no way this is the same shot one second+ later
            split_indices.append(i)

    shot_index_segments = []
    curr_index = 0
    for i in split_indices:
        shot_index_segments.append(shot_indicator_frames[curr_index:i])
        curr_index = i
    shot_index_segments.append(shot_indicator_frames[curr_index:])
    shot_peaks = [np.median(segment).astype(int) for segment in shot_index_segments]
    return shot_peaks

def get_shot_frames(df, peak):
    #lazy heuristic for now that is time-based, but easily customizable in future apps
    seconds_before = 2
    seconds_after = 2
    frames_before = int(seconds_before*frame_rate)
    total_frame_count = int((seconds_before+seconds_after)*frame_rate)
    start_frame = peak-frames_before
    end_frame = start_frame+total_frame_count
    shot_indices = pd.Index([int(i) for i in range(start_frame,end_frame)])
    return shot_indices

def plot_shot_frames(df):    
    #goal: I simply don't have enough compute power to use either unsupervized or supervised learning to detect shots, plus this is a tiny amount of data, so I will create a heuristic
    shot_peaks = get_shot_peaks(df)
    shotCount = 1
    for peak in shot_peaks:
        shot_indices = get_shot_frames(df, peak)
        plt.plot(shot_indices, df['wrist1']['y'][shot_indices], label="wrist1")
        plt.plot(shot_indices, df['wrist2']['y'][shot_indices], label="wrist2")
        #plt.scatter(shot_indicator_frames, wrist1_forehead_distance[shot_indicator_frames], label='WRIST_1_FILTERED')#, s=2)
        #plt.scatter(shot_indicator_frames, wrist2_forehead_distance[shot_indicator_frames], label='WRIST_2_FILTERED')#, s=2)
        plt.xlabel('Frame')
        plt.ylabel('Position')
        plt.title('Wrist position during shot '+str(shotCount))
        plt.legend()
        plt.show()
        shotCount+=1

plot_shot_frames(df)
