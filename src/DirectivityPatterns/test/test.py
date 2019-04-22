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

"""
data1 = data1.T[0,:]
data2 = data2.T[0,:]
data3 = data3.T[0,:]
"""

data = np.array([data1.astype(float)/32767, data2.astype(float)/32767, data3.astype(float)/32767])

x = Preprocessing(data,10).mixing()

y = FDICA(x, sample_freq=rate1,m_distance=10).fdica()

y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('./music1_2.wav', rate1, y[0])
wf.write('./music2_2.wav', rate2, y[1])
wf.write('./music3_2.wav', rate3, y[2])
