import os
import math
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from scipy.io import wavfile
from pprint import pprint

def block_audio(audio_input, sr=None, frame_size=2048, hop_ratio=0.5, pad=True):
    """
    Parameters
    ----------
    audio_input : np.ndarray or str
        Audio input. Should be able to come from either:
        - A NumPy array containing the audio signal.
        - A string path to an audio file (e.g., 'audio.wav').
    sr : int
        Sampling rate. Required if input is np.ndarray
    frame_size : int, optional
        Size of each frame in samples (default 2048).
    hop_ratio : float, optional
        Hop size as a ratio of the frame_size (default 0.5).
    pad : bool, optional
        If True (default), pads the signal with zeros to ensure all frames are the same length.
        If False, discards the last incomplete frame.
    Returns
    -------
    frames : np.ndarray
        2D array of shape (n_frames, frame_size).
    times : np.ndarray
        Array of start times (in seconds) for each frame.
    """

    # Handle input type
    if isinstance(audio_input, str):
        sr, audio_input = wavfile.read(audio_input)
    elif sr is None:
        raise ValueError("Must provide sampling rate with numpy array or file path as a string")

    # Convert to float in range [-1, 1)
    if audio_input.dtype == np.float32 or audio_input.dtype == np.float64:
        audio_input = audio_input
    else:
        # Determine bit depth and convert
        if audio_input.dtype == np.uint8:
            nbits = 8
        elif audio_input.dtype == np.int16:
            nbits = 16
        elif audio_input.dtype == np.int32:
            nbits = 32
        else:
            raise ValueError(f"Unsupported audio dtype: {audio_input.dtype}")

        audio_input = audio_input / float(2**(nbits - 1))

    # Convert to mono if stereo
    if len(audio_input.shape) > 1:
        audio_input = np.mean(audio_input, axis=1)

    # Calculate hop size as integer
    hop_size = int(hop_ratio * frame_size)

    # Calculate number of frames
    if pad:
        # Include all possible frames, padding the last one if necessary
        num_blocks = math.ceil((len(audio_input) - frame_size) / hop_size) + 1
        # Ensure we have at least one block even for very short audio
        num_blocks = max(1, num_blocks)
    else:
        # Only include complete frames
        num_blocks = max(0, (len(audio_input) - frame_size) // hop_size + 1)

    # Initialize output arrays
    audio_blocks = np.zeros([num_blocks, frame_size])

    # Compute time stamps
    times = (np.arange(0, num_blocks) * hop_size) / sr

    # Extract frames
    for n in range(num_blocks):
        i_start = n * hop_size
        i_stop = i_start + frame_size

        if i_stop <= len(audio_input):
            # Complete frame
            audio_blocks[n] = audio_input[i_start:i_stop]
        else:
            # Incomplete frame (only happens when pad=True)
            remaining_samples = len(audio_input) - i_start
            if remaining_samples > 0:
                audio_blocks[n, :remaining_samples] = audio_input[i_start:]
                # Rest of the frame is already zeros from initialization

    return audio_blocks, times


def estimate_f0(audio_frame, sr, minfreq=20, maxfreq=None, threshold=0.25):
  """
  Parameters
  ----------
  audio_frame : np.ndarray
      - A NumPy array containing the audio signal.
  sr : int
      Sampling rate. Required if input is np.ndarray
  minfreq : int, optional
      Minimum frequency in Hz that the ACF function will "look for"
  maxfreq : int, optional
      Hop size as a ratio of the frame_size (defaul sr/8).
  threshold : float, optional
      Minimum ACF value below which there is not high reliability of pitched content (default 0.25)
  Returns
  -------
  f0 : np.float64
      Fundamental frequency estimate between your minfreq and maxfreq parameters OR np.nan for a given frame of audio
  """
  # Handle inputs
  if maxfreq is None: maxfreq = sr / 8
  if maxfreq == 0: raise ValueError('Max Freq cannot be 0')
  if minfreq == 0: raise ValueError('Min Freq cannot be 0')
  # assign default output for later if no value is found
  f0 = np.nan
  # normalize from -1 to 1
  audio_frame = audio_frame / max(audio_frame)
  # apply periodic window function
  audio_frame = audio_frame * sig.windows.blackmanharris(len(audio_frame), sym=False)
  # subtract the mean
  sig.detrend(audio_frame, type='constant', overwrite_data=True)

  # convert frequencies to periods
  Tmax = 1/minfreq # lowest frequency becomes maximum period (in seconds)
  Tmin = 1/maxfreq # highest frequency becomes minimum period (in seconds)
  Nmax = int(np.ceil(Tmax * sr)) # seconds * samples/seconds = samples
  Nmin = int(np.floor(Tmin * sr))
  # calculate acf
  corr = np.correlate(audio_frame, audio_frame, mode='full')
  corr = corr[len(audio_frame)-1:]
  corr = corr / corr[0]


  corr = corr[Nmin:Nmax+1] # cut off periods above Nmax

  # find strongest peak
  peak_indeces, props = sig.find_peaks(corr, height=threshold, distance=Nmin)
  if len(peak_indeces) != 0:
    strongest_peak = peak_indeces[np.argmax(props["peak_heights"])]
    # convert to frequency in Hz
    k = Nmin + strongest_peak
    f0 = sr / k
    # pprint(np.argsort(-props["peak_heights"]))
  return f0


audio1 = 'pansoori_female.wav'
audio2 = 'violin-sanidha.wav'

input = audio2

#get the sr
sr, input = wavfile.read(input)

#run blocking
data, times = block_audio(input, sr=sr, frame_size=2048, hop_ratio=0.5, pad=True)

#apply auto correlation
f0s = [estimate_f0(frame, sr, 100, 800, threshold = 0.9) for frame in data]
f0s = np.array(f0s, dtype=np.float64)


#plot the contour
plt.plot(times, f0s, marker='.')
plt.xlabel("Time (s)")
plt.ylabel("f0 (Hz)")
plt.title("Pitch contour over time")
plt.ylim(100, 800)  # focus on expected pitch range
plt.grid(True)
plt.show()