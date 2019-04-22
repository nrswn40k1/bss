#音声信号の行列xの形は(3,44100)
#音は10秒

#半径1メートルの半円上に音源が30,90,150度にあり、観測者は端および中心の３点にいることを想定する
#中心からはそれぞれ１メートルであるが、端からは長い方は1.9319メートル、短い方は0.51763メートルである

import numpy as np
from scipy.signal import stft, istft

class Preprocessing():

    def __init__(self,s,r):
        '''
        sは音源信号行列((3,441000))
        rは音源が並んでいる円の半径(スカラー)
        '''
        self.s = s
        self.r = r

    def mixing(self):
        all2bDis = self.r
        a2cDis = self.r * np.sqrt(2+np.sqrt(2))
        a2aDis = self.r * np.sqrt(2-np.sqrt(2))
        b2aDis = self.r * np.sqrt(2)

        all2bTime = all2bDis/340.5 #これが基準
        a2cTime = a2cDis/340.5
        a2aTime = a2aDis/340.5
        b2aTime = b2aDis/340.

        F,_,S = stft(self.s, 44100, "boxcar", 256, 128)
        n_bin = len(F)
        X = np.array(S.shape,dtype='complex64')
        for f in range(n_bin):
            X[0,f,:] = S[0,f,:]*np.exp(-1j*2*np.pi*F[f]*a2aTime)+S[1,f,:]*np.exp(-1j*2*np.pi*F[f]*b2aTime)+S[2,f,:]*np.exp(-1j*2*np.pi*F[f]*a2cTime)
            X[1,f,:] = S[0,f,:]*np.exp(-1j*2*np.pi*F[f]*all2bTime)+S[1,f,:]*np.exp(-1j*2*np.pi*F[f]*all2bTime)+S[2,f,:]*np.exp(-1j*2*np.pi*F[f]*all2bTime)
            X[2,f,:] = S[0,f,:]*np.exp(-1j*2*np.pi*F[f]*a2cTime)+S[1,f,:]*np.exp(-1j*2*np.pi*F[f]*b2aTime)+S[2,f,:]*np.exp(-1j*2*np.pi*F[f]*a2aTime)

        _, x = istft(X, 44100, "boxcar", 256, 128)
        return x



