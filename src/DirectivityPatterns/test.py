import numpy as np
import sys
import scipy.io.wavfile as wf
from DirectivityPatterns import ICA, FDICA
from sound_mixing import Preprocessing

#prepare data
rate1, data1 = wf.read('./fanfare.wav')
rate2, data2 = wf.read('./loop1.wav')
rate3, data3 = wf.read('./strings.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')

data = [data1.astype(float)/32768, data2.astype(float)/32768, data3.astype(float)/32768]

x = Preprocessing(data,10).mixing()

wf.write('./music_mix1.wav', rate1, x[0])
wf.write('./music_mix2.wav', rate2, x[1])
wf.write('./music_mix3.wav', rate3, x[2])

y = FDICA(x, sample_freq=rate1,m_distance=10).fdica()

y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('./music1.wav', rate1, y[0])
wf.write('./music2.wav', rate2, y[1])
wf.write('./music3.wav', rate3, y[2])
