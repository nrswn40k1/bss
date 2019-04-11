import numpy as np
import sys
import scipy.io.wavfile as wf
from DirectivityPatterns import ICA, FDICA
from sound_mixing import Preprocessing

#prepare data
rate1, data1 = wf.read('./raw_1.wav')
rate2, data2 = wf.read('./raw_2.wav')
rate3, data3 = wf.read('./raw_3.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')

new_data1 = data1.T[0,:]
new_data2 = data2.T[0,:]
new_data3 = data3.T[0,:]

sys.exit()

data = np.array([new_data1.astype(float)/32678, new_data2.astype(float)/32678, new_data3.astype(float)/32678])
print(data.shape)
x = Preprocessing(data,100).mixing()

y = FDICA(x, sample_freq=rate1,m_distance=100).fdica()

y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('./music1.wav', rate1, y[0])
wf.write('./music2.wav', rate2, y[1])
wf.write('./music3.wav', rate3, y[2])
