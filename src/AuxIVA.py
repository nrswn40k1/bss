import numpy as np
from numpy.linalg import inv
from scipy.signal import stft, istft


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
        self.max_iter = 10
        self.eta = 2.5 * 10 ** (-2)  # is step size
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
        I = np.eye(nchannel, dtype='complex64')
        W = np.zeros((n_bin,nchannel,nchannel),dtype='complex64')
        for f in range(n_bin):
            W[f,:,:] = I

        for i in range(self.max_iter):
            for f in range(n_bin):
                for k in range(nchannel):
                    r = self.Y_L2norm(W[f,:,:],X[:,f,:],k,n_bin,n_timesegment)
                    fai = self.WeigtingFunction(r,self.beta,n_timesegment)
                    V = self.CovarianceMatrix(fai,X[:,f,:],n_timesegment)
                    W[f,k,:] = np.dot(np.linalg.inv(np.dot(W[f,:,:],V)),np.eye(nchannel)[k].T)
                    W[f,k,:] = W[f,k,:]/np.sqrt(np.dot(np.dot(np.conjugate(W[f,k,:].T),V),W[f,k,:]))
                
                Y = np.dot(W[f,:,:],X[:,f,:])
                c = 0
                for k in range(nchannel):
                    for t in range(n_timesegment):
                        c += np.absolute(Y[k,t])**2
                c = c/n_timesegment/nchannel/n_bin
                W[f,:,:] = W[f,:,:]/np.linalg.norm(c)
            print(i)
        return W




    def Y_L2norm(self,W,X,k,n_bin,n_timesegment):
        r = np.zeros((n_timesegment))
        for i in range(n_bin):
            r += np.power(np.absolute(np.dot(np.conjugate(W[k].T),X)),2)
        return np.sqrt(r)
    
    def WeigtingFunction(self,r,beta,n_timesegment,fai0=1000):
        for t in range(n_timesegment):
            if(r[t]>=fai0):
                r[t] = r[t]**(beta-2)
            else:
                r[t]=fai0
        return r

    def CovarianceMatrix(self,fai,X,n_timesegment):
        nchannel = self.nchannel
        V = np.zeros((nchannel,nchannel), dtype='complex64')
        for i in range(n_timesegment):
            V += fai[i]*np.dot(X[:,np.newaxis,i],np.conjugate(X[:,np.newaxis,i].T))
        return V/n_timesegment
    