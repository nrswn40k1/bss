#音声信号の行列xの形は(3,44100)
#音は10秒

#半径1メートルの半円上に音源が30,90,150度にあり、観測者は端および中心の３点にいることを想定する
#中心からはそれぞれ１メートルであるが、端からは長い方は1.9319メートル、短い方は0.51763メートルである

import numpy as np

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
        a2cDis = self.r * np.sqrt(2+np.sqrt(3))
        a2aDis = self.r * np.sqrt(2-np.sqrt(3))
        b2aDis = self.r * np.sqrt(2)

        all2bTime = all2bDis/340.5 #これが基準
        a2cTime = a2cDis/340.5
        a2aTime = a2aDis/340.5
        b2aTime = b2aDis/340.5

        indexA2A = int(np.round((a2aTime - all2bTime)*44100))
        indexB2A = int(np.round((b2aTime - all2bTime)*44100))
        indexC2A = int(np.round((a2cTime - all2bTime)*44100))
        

        s = self.s[:][50000:391000]
        x = np.zeros((3,341000), dtype=np.float32)

        x[0][:] = self.s[0][(50000+indexA2A):(391000+indexA2A)]+self.s[1][(50000+indexB2A):(391000+indexB2A)]+self.s[2][(50000+indexC2A):(391000+indexC2A)]
        x[1][:] = self.s[0][50000:391000] + self.s[1][50000:391000] + self.s[2][50000:391000]
        x[2][:] = self.s[0][(50000+indexC2A):(391000+indexC2A)]+self.s[1][(50000+indexB2A):(391000+indexB2A)]+self.s[2][(50000+indexA2A):(391000+indexA2A)]
        
        return x



