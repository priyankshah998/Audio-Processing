# import seaborn
import numpy as np
import scipy
import librosa, librosa.display
import matplotlib.pyplot as plt


data, sr = librosa.load('b1.wav', sr=44100)
bin_per_octave = 36
hop_length = 512
onset_samples = librosa.onset.onset_detect(data,
                                           sr=sr, units='samples', 
                                           hop_length=hop_length, 
                                           backtrack=False,
                                           pre_max=20,
                                           post_max=20,
                                           pre_avg=100,
                                           post_avg=100,
                                           delta=0.2,
                                           wait=0)
# onset_boundaries = np.concatenate([[0], onset_samples, [len(data)]])
onset_samples = np.concatenate([[0], onset_samples])
print(onset_samples)
onset_times = librosa.samples_to_time(onset_samples, sr=sr)
plt.figure(figsize=(12, 5))
librosa.display.waveplot(data, sr)
plt.title('Onset data')
plt.vlines(onset_times, -0.5, 0.5, color='r', alpha=0.8)
plt.show()
print(onset_times)


def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
    
    # Compute autocorrelation of input segment.
    r = librosa.autocorrelate(segment)
    
    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
    
    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr)/i
    return f0


y = [estimate_pitch(data[onset_samples[i]:onset_samples[i+1]], sr)for i in range(len(onset_samples)-1)]
print(y)
note = librosa.hz_to_note(y)
print(note)





