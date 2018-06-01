import matplotlib.pyplot as plt
import numpy as np
import librosa
import peakutils
import librosa.display as ld

# Load the audio file
data, sr = librosa.load('sample_piano.wav', sr=44100)
n_fft = 1024
hop_length = 512

# Detect onsets
onset_samples = librosa.onset.onset_detect(data, sr=sr, units='samples', hop_length=hop_length, backtrack=False)
onset_samples = np.concatenate([[0], onset_samples, [len(data)]])

# Convert onsets from samples to time
onset_times = librosa.samples_to_time(onset_samples, sr=sr)

# print(onset_times)
notes = []

# Determine the note  and its frequency
for i in range(len(onset_samples)):
    if i is len(onset_samples)-1:
        break
    sample_data = data[onset_samples[i]:onset_samples[i+1]]
    # Compute stft of each onsets detected
    sft = librosa.stft(sample_data, n_fft=n_fft, hop_length=hop_length)
    f = []
    for x in range(len(sft[0, :])):
        F = sft[:, x]
        indexes = peakutils.indexes(abs(F))
        if len(indexes) > 0:
            f.append((indexes[0] * sr)/n_fft)
    # print(f)
    avg_freq = sum(f)/len(f)
    notes.append(librosa.hz_to_note(avg_freq))


# print(notes)
print('No of onsets = ' + str(len(onset_samples)))
print('No of notes = ' + str(len(notes)))

final = []
# Calculate each notes time duration
for i in range(len(onset_times)-1):
    diff = onset_times[i+1] - onset_times[i]
    final.append((notes[i], diff))

# Final output as a tuple of the detected note and its duration in secs
print(final)

# Visualize all the onsets detected
plt.figure(figsize=(12, 5))
ld.waveplot(data, sr)
plt.title('Onset data')
plt.vlines(onset_times, -0.5, 0.5, color='r', alpha=0.8)
plt.show()

