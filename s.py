import numpy as np
from numpy.fft import rfft
from scipy.io import wavfile
from scipy import signal
from scipy.signal import blackmanharris, fftconvolve
import matplotlib.pyplot as plt

from numpy import argmax, mean, diff, log
from parabolic import parabolic


filename = 'sample_piano.wav'
# downloaded from http://vocaroo.com/i/s1KZzNZLtg3c

# Parameters
time_start = 0  # seconds
time_end = 1  # seconds
filter_stop_freq = 70  # Hz
filter_pass_freq = 100  # Hz
filter_order = 1001

# Load data
fs, audio = wavfile.read(filename)
audio = audio.astype(float)

# High-pass filter
nyquist_rate = fs / 2.
desired = (0, 0, 1, 1)
bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

# Examine our high pass filter
w, h = signal.freqz(filter_coefs)
f = w / 2 / np.pi * fs  # convert radians/sample to cycles/second
plt.plot(f, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [Hz]')
plt.xlim((0, 300))

audio = audio[:,0]
# Apply high-pass filter
filtered_audio = signal.filtfilt(filter_coefs, [1], audio)

# Only analyze the audio between time_start and time_end
time_seconds = np.arange(filtered_audio.size, dtype=float) / fs
audio_to_analyze = filtered_audio[(time_seconds >= time_start) &
                                  (time_seconds <= time_end)]


def freq_from_HPS(sig, fs):
    """
    Estimate frequency using harmonic product spectrum (HPS)

    """
    windowed = sig * blackmanharris(len(sig))

    from pylab import subplot, plot, log, copy, show

    # harmonic product spectrum:
    c = abs(rfft(windowed))
    maxharms = 8
    subplot(maxharms, 1, 1)
    plot(log(c))
    for x in range(2, maxharms):
        a = copy(c[::x])  # Should average or maximum instead of decimating
        # max(c[::x],c[1::x],c[2::x],...)
        c = c[:len(a)]
        i = argmax(abs(c))
        true_i = parabolic(abs(c), i)[0]
        print('Pass %d: %f Hz'% (x, fs * true_i / len(windowed)))
        c *= a
        subplot(maxharms, 1, x)
        plot(log(c))
    show()


fundamental_frequency = freq_from_HPS(audio_to_analyze, fs)
print('Fundamental frequency is {} Hz').format(fundamental_frequency)