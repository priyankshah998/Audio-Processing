from numpy.fft import fft, ifft
import math
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming  

index1=15000;
frameSize=256;
spf = wave.open('a1.wav','r');
fs = spf.getframerate();
signal = spf.readframes(-1);
signal = np.fromstring(signal, 'Int16');
index2=index1+frameSize-1;
frames=signal[index1:int(index2)+1]

zeroPaddedFrameSize=16*frameSize;

frames2=frames*hamming(len(frames));   
frameSize=len(frames);

if (zeroPaddedFrameSize>frameSize):
    zrs= np.zeros(zeroPaddedFrameSize-frameSize);
    frames2=np.concatenate((frames2, zrs), axis=0)

fftResult=np.log(abs(fft(frames2)));
ceps=ifft(fftResult);

posmax = ceps.argmax();

result = fs/zeroPaddedFrameSize*(posmax-1)

print(result)