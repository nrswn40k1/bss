# bss
## 工学博覧会 : 音源分離チーム

### FDICA

#### Requirements
libraries
- munkres
- tqdm
- numpy
- scipy

#### Usage
First, change the current directory to src.
```
pip install tqdm
pip install munkres
cd src
```
Second, for instance, 

```
import numpy as np
import scipy.io.wavfile as wf
from FDICA import ICA, FDICA

#prepare data
rate1, data1 = wf.read('./mix_1.wav')
rate2, data2 = wf.read('./mix_2.wav')
rate3, data3 = wf.read('./mix_3.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')

data = [data1.astype(float), data2.astype(float), data3.astype(float)]


y = FDICA(data, sample_freq=rate1).fdica()

y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('./music1.wav', rate1, y[0])
wf.write('./music2.wav', rate2, y[1])
wf.write('./music3.wav', rate3, y[2])
```

#### Reference
- Evaluation of blind signal separation method using directivity pattern under reverberant condition
- An Approach to Blind Source Separation. Based on Temporal Structure of Speech Signals. 

