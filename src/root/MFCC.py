from librosa.feature import mfcc
import scipy
from scipy import io
from scipy.io import wavfile
import glob
import numpy as np
import os

def write_ceps(ceps,fn):
    base_fn,ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn,ceps)

def create_ceps(fn):
    sample_rate,X = io.wavfile.read(fn)
    ceps,mspec,spec = mfcc(X)
    write_ceps(ceps,fn)

if __name__ == "__main__":
    #file name を入れてください
    create_ceps()
