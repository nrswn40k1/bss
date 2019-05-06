# BSS (Blind Source Separation)
## 工学博覧会2019 : 音声処理班音源分離チーム

### FDICA
-------------------------------
FDICA is frequency domain independent component analysis. 

#### Requirements (library dependency)
You need Python 3.4 or later to run FDICA.
- munkres
- tqdm
- numpy
- scipy

#### Quick start
First, install libraries and change the current directory to src.
```
pip install numpy
pip install scipy
pip install tqdm
pip install munkres
cd src
```
Second, for instance,

```python
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

#### Usage
You can choose three different fai function.

#### Reference
- Evaluation of blind signal separation method using directivity pattern under reverberant condition
- An Approach to Blind Source Separation Based on Temporal Structure of Speech Signals. 


### IVA
-------------------------------
IVA is independent vector analysis.

#### Requirements (library dependency)
You need Python 3.6 or later to run IVA.
- tqdm
- numpy
- scipy

#### Quick start
First, install libraries and change the current directory to src.
```
cd src
```
Second, for instance,

```python
import numpy as np
import cis
from IVA import IVA

rate1, data1 = cis.wavread('./samples/mixdata/mix1.wav')
rate2, data2 = cis.wavread('./samples/mixdata/mix2.wav')
rate3, data3 = cis.wavread('./samples/mixdata/mix3.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')
fs = rate1
x = np.array([data1, data2, data3], dtype=np.float32)
y = IVA(x, fs).iva()

cis.wavwrite('./samples/sepdata/IVA/music1_r.wav', fs, y[0])
cis.wavwrite('./samples/sepdata/IVA/music2_r.wav', fs, y[1])
cis.wavwrite('./samples/sepdata/IVA/music3_r.wav', fs, y[2])
```

#### Reference
- Blind Source Separation Exploiting Higher-Order Frequency Dependencies

### ILRMA
-------------------------------
ILRMA is Independent Low-Rank Matrix Analysis.

#### Requirements (library dependency)
You need Python 3.6 or later to run ILRMA.
- tqdm
- numpy
- scipy

#### Quick start
First, install libraries and change the current directory to src.
```
cd src
```
Second, for instance,

```python
import numpy as np
import cis
from ILRMA import ILRMA

rate1, data1 = cis.wavread('./samples/mixdata/mix1.wav')
rate2, data2 = cis.wavread('./samples/mixdata/mix2.wav')
rate3, data3 = cis.wavread('./samples/mixdata/mix3.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')
fs = rate1
x = np.array([data1, data2, data3], dtype=np.float32)
y = ILRMA(x, fs, L=2).ilrma()       # L is # of bases for each source

cis.wavwrite('./samples/sepdata/ilrma_1.wav', fs, y[0])
cis.wavwrite('./samples/sepdata/ilrma_2.wav', fs, y[1])
cis.wavwrite('./samples/sepdata/ilrma_3.wav', fs, y[2])
```

#### Reference
- Blind Source Separation Based on Independent Low-Rank Matrix Analysis