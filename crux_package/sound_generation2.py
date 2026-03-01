import numpy as np
import warnings
from pathlib import Path
from scipy.io import wavfile
from .utils import next_valid
import pickle

#defaults
sampling_rate = 10000
soa = 1 * sampling_rate

def generate_gaussian(length, scaling=16, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    mean = 0
    std = 1 / (scaling * np.sqrt(2))
    noise = np.random.normal(mean, std, length)

    return noise
################
def generate_pulse(freq, rel_amp, samples, scaling=16):
    #give freq in oscillations per index
    freq = np.asarray(freq, dtype=float)
    rel_amp = np.asarray(rel_amp, dtype=float)

    if freq.ndim != 1 or rel_amp.ndim != 1:
        raise ValueError("freq and rel_amp must be 1D")
    if len(freq) != len(rel_amp):
        raise ValueError("freq and rel_amp must have the same length")

    # Sample index
    n = np.arange(samples)

    # Sum sine waves
    pulse = np.zeros(samples, dtype=float)
    for f, a in zip(freq, rel_amp):
        pulse += a * np.sin(2 * np.pi * f * n)

    # Normalization placeholder (per instructions)
    norm = 1
    pulse = pulse / norm

    # Apply scaling and clip
    pulse = pulse / scaling
    pulse = np.clip(pulse, -1.0, 1.0)

    return pulse
######################
# generates an envelope with range [0, 1]
def generate_envelope(length, start_length, end_length, start_degree, end_degree):
    if start_length + end_length > length:
        warnings.warn(
            "Fade length is longer than pulse length. Warning.",
            RuntimeWarning
        )
    envelope = np.ones(length)
    if start_degree >0:
        start_fade = 1-np.abs((np.linspace(-1,0,start_length))**start_degree)
    elif start_degree<0:
        start_fade = np.abs((np.linspace(0,1,start_length))**(-start_degree))
    else:
        start_fade = np.ones(start_length)
    if end_degree > 0:
        end_fade = 1-np.abs((np.linspace(0,1,end_length))**end_degree)
    elif end_degree < 0:
        end_fade = np.abs((np.linspace(-1,0,end_length))**(-end_degree))
    else:
        end_fade = np.ones(end_length)


    envelope[0:start_length] = start_fade
    envelope[length-end_length:length] = end_fade
    return envelope
###########################
def generate_train(pulses, counts, soa, shuffle=False, random_state=None):
#take in 2D matrix of pulses -> (frequency,length) pairs (units = indices)
#take soa -> in indices
#take counts -> 1D matrix, # of occurrences of each frequency
#shuffle
    if random_state is not None:
        np.random.seed(random_state)

    soa_indices = soa
    pulse_number = pulses.shape[0]
    pulse_list = []
    for i in range(pulse_number):
        pulse_list.append(i)
        if len(pulses[i,:]) > soa:
            print(
                f"WARNING WARNING WARNING: Pulse {i} has greater length than SOA."
            )
            
    pulse_list = np.array(pulse_list)

    train_blocks = np.repeat(pulse_list, counts)
    if shuffle:
        np.random.shuffle(train_blocks)

    click_count = np.sum(counts)


    train = np.zeros(soa_indices*click_count)
    for n in range(len(train_blocks)):
        cart = np.zeros(soa_indices)
        which= train_blocks[n]
        signal_mini = pulses[which,:]
        length = len(signal_mini)
        cart[:length] = signal_mini
        train[soa*n:soa*n+soa_indices] = cart

    staggers = np.zeros(click_count)
    starts = np.arange(click_count) * soa + staggers
    lengths = np.array([len(pulses[train_blocks[n]]) for n in range(click_count)])
    ends = starts + lengths

    datadict = {"sampling_rate": sampling_rate, "soa": soa, "click_count": click_count, "starts": starts, "lengths": lengths, "ends": ends}

    return train, train_blocks, datadict
#######################
def generate_train_with_stagger(pulses, counts, soa, shuffle=False, random_state=None):
#take in 2D matrix of pulses -> (frequency,length) pairs (units = indices)
#take soa -> in indices
#take counts -> 1D matrix, # of occurrences of each frequency
#shuffle
    if random_state is not None:
        np.random.seed(random_state)

    soa_indices = soa
    pulse_number = pulses.shape[0]
    pulse_list = []
    max_stagger_beats = np.zeros(pulse_number)
    for i in range(pulse_number):
        pulse_list.append(i)
        if len(pulses[i,:]) > soa:
            print(
                f"WARNING WARNING WARNING: Pulse {i} has greater length than SOA."
            )
        max_stagger_beats[i] = np.floor(250*(soa - len(pulses[i,:]))/sampling_rate)-25#25 to be changed ... how much minimum pause? (25 stagger beat = 1000 sample = 0.1s)
            
    pulse_list = np.array(pulse_list)

    train_blocks = np.repeat(pulse_list, counts)
    if shuffle:
        np.random.shuffle(train_blocks)

    click_count = np.sum(counts)
    staggers = np.zeros(click_count)
    train = np.zeros(soa_indices*click_count)
    for n in range(len(train_blocks)):
        cart = np.zeros(soa_indices)
        which= train_blocks[n]
        j = np.random.randint(0,max_stagger_beats[which]+1)
        delay = int(j * sampling_rate/250)
        signal_mini = pulses[which,:]
        length = len(signal_mini)
        cart[delay:delay+length] = signal_mini
        train[soa*n:soa*n+soa_indices] = cart
        staggers[n] = delay

    starts = np.arange(click_count) * soa + staggers
    lengths = np.array([len(pulses[train_blocks[n]]) for n in range(click_count)])
    ends = starts + lengths

    datadict = {"sampling_rate": sampling_rate, "soa": soa, "click_count": click_count, "starts": starts, "lengths": lengths, "ends": ends}

    return train, train_blocks, datadict
#######################
def array_to_wav(arr, sampling_rate, save_path=None):
    try:
        # --- Basic validation ---
        arr = np.asarray(arr)

        if arr.ndim != 1:
            raise ValueError("Input array must be 1D")
        if not np.issubdtype(arr.dtype, np.floating):
            raise TypeError("Input array must be floating-point")

        # --- Range check ---
        if np.any(arr < -1.0) or np.any(arr > 1.0):
            warnings.warn(
                "Input array contains values outside [-1, 1]. "
                "Clipping will occur.",
                RuntimeWarning
            )
            arr = np.clip(arr, -1.0, 1.0)

        # --- Convert to 24-bit PCM ---
        # 24-bit PCM uses int32 container with lower 8 bits unused
        max_int32 = 2**31 - 1
        pcm32 = (arr * max_int32).astype(np.int32)

        if save_path is None:
            john = next_valid("output",".wav")
            save_path = Path(john)
        else:
            john = save_path
            save_path = Path(save_path)

        wavfile.write(save_path, int(sampling_rate), pcm32)
        print(f"Audio file saved as: {save_path}")

        return john
    
    except Exception as e:
        warnings.warn(f"Failed to write WAV file: {e}", RuntimeWarning)
        return False
###############################
def sound_data(data, save_path=None):
    if save_path is None:
        john = next_valid("output",".pkl")
        save_path = Path(john)
    else:
        save_path = Path(save_path)
    with open(save_path, "wb") as file:
        pickle.dump(data, file)
    print(f"data file saved as: {save_path}")

"""
generate train will produce 3 variables - 

train: returns wave itself (put through array_to_wav)

train_blocks: returns order of pulses played

datadict: includes all data relevant for analysis (put through sound_data)


"""
