import numpy as np
from numpy.linalg import inv
from scipy.signal import stft, istft
import sys


# suppose that the number of sources and microphones are equal.

# M : # of channels whose index is m
# K : # of frequency bins whose index is k
# T : # of time frames whose index is t

class AuxIVA:

    def __init__(self, x, sample_freq, beta=0.2, win='hanning', nperseg=256, noverlap=128,nchannel=3):
        '''
        @param(win):str, desired window to use.
        @param(nperseg): length of each segment.
        @param(noverlap): number of points to overlap between segments.
        '''
        self.max_iter = 100
        self.x = np.array(x)
        self.sample_freq = sample_freq
        self.win = win
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nchannel = nchannel
        self.beta = beta

    def auxiva(self):
        '''
        X is complex64-type-3-dementional array whose x axis is microphie , y axis is the segment times, z is frequency respectively.
        @output(x_prd): 2 dimensional array whose 1st axis is the source index, 2nd is data of them.
        '''

        f, _, X = stft(self.x, self.sample_freq, self.win, self.nperseg, self.noverlap)
        # X is (channel index, freq index, time segment index)
        n_bin = len(f)
        n_timesegment = len(X[0,0,:])
        W = self.W_estimate(X,n_bin,n_timesegment)
        Y = np.zeros(X.shape,dtype='complex64')
        for f in range(n_bin):
            Y[:,f,:] = np.dot(W[f,:,:],X[:,f,:])

        _, x_prd = istft(Y, self.sample_freq, self.win, self.nperseg, self.noverlap)

        return x_prd



    def W_estimate(self,X,n_bin,n_timesegment):
        nchannel = self.nchannel
        W = np.zeros((n_bin,nchannel,nchannel),dtype='complex64')
        for f in range(n_bin):
            W[f,:,:] = np.eye(nchannel, dtype='complex64')

        for i in range(self.max_iter):
            for f in range(n_bin):
                r = self.Y_L2norm(W[f,:,:],X[:,f,:],n_bin,n_timesegment,nchannel)
                fai = self.WeigtingFunction(r,self.beta,n_timesegment)
                V = self.CovarianceMatrix(fai,X[:,f,:],n_timesegment)
                for k in range(nchannel):
                    w_k = np.zeros(nchannel,dtype="complex64")
                    w_k = np.dot(np.linalg.inv(np.dot(W[f,:,:],V[k,:,:])),np.eye(nchannel)[k])
                    w_k = w_k/np.sqrt(np.dot(np.dot(np.conjugate(w_k.T),V[k,:,:]),w_k))
                    W[f,k,:] = np.conjugate(w_k)
                Y = np.dot(W[f,:,:],X[:,f,:])
                Y_2 = np.power(Y,2)
                c = np.sum(Y_2)
                c = c/n_timesegment/nchannel/n_bin
                W[f,:,:] = W[f,:,:]/np.linalg.norm(c)
            print(i)
        return W

    def Y_L2norm(self,W,X,n_bin,n_timesegment,nchannel):
        r = np.zeros((nchannel,n_timesegment))
        for i in range(n_bin):
            r += np.power(np.absolute(np.dot(W,X)),2)
        return np.sqrt(r)
    
    def WeigtingFunction(self,r,beta,n_timesegment,fai0=1000):
        r = np.power(r,beta-2)
        return np.where(fai0<r,fai0,r)

    def CovarianceMatrix(self,fai,X,n_timesegment):
        nchannel = self.nchannel
        V = np.zeros((nchannel,nchannel,nchannel), dtype='complex64')
        for k in range(nchannel):
            V[k,:,:] = np.dot(fai[k]*X,np.conjugate(X.T))
        return V/n_timesegment
    