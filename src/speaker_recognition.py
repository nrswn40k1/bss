import pickle
import numpy as np
from librosa.feature import mfcc
from librosa import load
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
import sys
import scipy.io.wavfile as wf

class speaker_recognition:
    def __init__(self, n_people):
        self.n_people = n_people
        self.sc = StandardScaler()
        self.svc = pickle.load(open("./group/model.sav","rb"))
        self.sc = pickle.load(open("./group/sc.sav","rb"))

    def create_ceps(self,fn):
        y,sr = load(fn)
        ceps = mfcc(y,sr)
        ceps = ceps.T
        return ceps

    def predict(self,ceps):
        ceps = self.sc.transform(ceps)
        y = self.svc.predict(ceps)
        count = np.bincount(y)
        mode = np.argmax(count)
        return mode

    def transform(self):
        index = np.zeros(self.n_people)
        for i in range(self.n_people):
            fn = "./group/output_%d.wav"%i
            ceps = self.create_ceps(fn)
            index[i] = self.predict(ceps)
        return index
        

if __name__=="__main__":
    n_people = 3
    SR = speaker_recognition(n_people)
    index = SR.transform()
    _,data = wf.read("./group/output_%d.wav"%index[0])
    x = np.array(data)
    for i in range(1,n_people):
        _, data = wf.read('./group/output_%d.wav'%index[i])
        x = np.vstack([x,data.astype(float)])

    
    print(x.shape)
