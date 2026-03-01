import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# gets a single channel (e.g. "EXG Channel 0") from a .txt file and returns as np array
def get_channel_from_txt(file, channel):
    df = pd.read_csv(file, sep=', ', comment="%", engine='python')
    values = np.array(df.loc[:, channel])
    return values


# checks if any samples were not recieved from the hardware side and fills missing samples
# by linearly interpolating between the two neighboring samples
def interpolate_missing_samples(sample_indices, voltages, cycle_size=256):
    sample_indices = sample_indices.astype(int)

    if sample_indices.shape != voltages.shape:
        raise ValueError("sample_indices and voltages must have same shape")

    sample_indices = np.asarray(sample_indices)
    voltages = np.asarray(voltages)

    output_voltages = []

    last_idx = None
    last_v = None

    for idx, v in zip(sample_indices, voltages):

        if last_idx is None:
            output_voltages.append(v)
            last_idx = idx
            last_v = v
            continue

        # Detect wraparound (new cycle)
        if idx <= last_idx:
            # Fill remaining indices in previous cycle with last value
            for _ in range(last_idx + 1, cycle_size):
                output_voltages.append(last_v)

            # Fill missing at start of new cycle with current value
            for _ in range(0, idx):
                output_voltages.append(v)

            output_voltages.append(v)

        else:
            gap = idx - last_idx

            if gap > 1:
                # Linear interpolation for missing internal samples
                steps = np.linspace(last_v, v, gap + 1)[1:-1]
                output_voltages.extend(steps)

            output_voltages.append(v)

        last_idx = idx
        last_v = v

    return np.array(output_voltages)


# applies a bandpass filter to data
# fs is the sampling rate
# lowcut and highcut are the minimum and maximum frequencies
def bandpass(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data




#subarrays of every click response
def get_click_responses(signal, click_idx, end_idx):
    out = []
    for n in range(len(click_idx)):
        out.append(signal[click_idx[n]:end_idx[n]])
    return np.stack(out)
    
#subarrays of the windows between two click responses (first occurs after first click)
def get_silence_responses(signal, click_idx, end_idx):
    out = []
    for n in range(len(click_idx)-1):
        out.append(signal[end_idx[n]:click_idx[n+1]])
        if n == len(click_idx)-1:
           length = click_idx[n+1] - end_idx[n]
           out.append(signal[end_idx[n+1]:end_idx[n+1]+length])

    return out #NOT an array
