import numpy as np
import scipy.io.wavfile as wf
from FDICA import ICA, FDICA

#prepare data
rate1, data1 = wf.read('./fanfare.wav')
rate2, data2 = wf.read('./loop1.wav')
rate3, data3 = wf.read('./strings.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')

data = [data1.astype(float), data2.astype(float), data3.astype(float)]

y = FDICA(data, sample_freq=rate1).fdica()

y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('./music1.wav', rate1, y[0])
wf.write('./music2.wav', rate2, y[1])
wf.write('./music3.wav', rate3, y[2])