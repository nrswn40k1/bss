import numpy as np
from numpy.linalg import inv
from scipy.signal import stft, istft
from munkres import Munkres, print_matrix
from tqdm import tqdm


#suppose that the number of sources and microphones are equal.

class ICA:
    
    def __init__(self):
        self.max_iter = 50
        self.eta = 1.0 * 10 ** (-4) # is step size

    def ica(self, x):
        x = np.array(x)
        w = self.__optimize(x)
        y = np.dot(w, x)
        return y, w

    def __fai_func(self, y):
        return 1/(1+np.exp(-y.real)) + 1j*1/(1+np.exp(-y.imag))

    def __alpha(self, y):
        return  np.dot(self.__fai_func(y), y.T.conjugate())    

    def __optimize(self, x):
        r,c = x.shape
        w = np.zeros((r,r), dtype=np.complex64)
        w += np.diag(np.ones(r))
        
        
        for _ in range(self.max_iter):
            y = np.dot(w, x)
            alpha = np.zeros((r,r), dtype=np.complex64)

            for i in range(c):
                alpha += self.__alpha(y[:,i])
            
            alpha = alpha/c
            w += self.eta * np.dot((np.diag(np.diag(alpha)) - alpha),  inv(w.T.conjugate()))
            
        return w
    
class FDICA(ICA):
    '''
    The class FDCIA is inherited from ICA
    '''

    def __init__(self, x, sample_freq, win='boxcar', nperseg=256, noverlap=128):
        '''
        @param(d): is a vector which represents the distance from a reference point.
        @param(win):str, desired window to use.
        @param(nperseg): length of each segment.
        @param(noverlap): number of points to overlap between segments.
        '''
        super().__init__()
        self.m_shit = 5
        self.x = np.array(x)
        self.sample_freq = sample_freq
        self.win = win
        self.nperseg = nperseg
        self.noverlap = noverlap

    def fdica(self):
        '''
        X is complex64-type-3-dementional array whose x axis is microphie , y axis is the segment times, z is frequency respectively.
        @output(x_prd): 3 dimensional array whose 1st axis is the source index, 2nd is the microphon index, third is data of them.
        '''

        f,_,X = stft(self.x, self.sample_freq, self.win, self.nperseg, self.noverlap)
        # X is (channel index, freq index, time segment idex)

        y = self.reconstruct(f,X)

        _,x_prd = istft(y[:,:,:,0], self.sample_freq, self.win, self.nperseg, self.noverlap)
        
        return x_prd


    def reconstruct(self,f,X):
        '''
        This func is the way of permutation.
        @param(f): frequency array.
        @param(X): stft of time series x. 
        @output(y): 
        v is 4 dementional array whose 1st axis is the source index, 2nd axis is the microphone index, 4th axis is frequency index.
        '''
        
        epsilon_v = np.zeros(X.shape)

        v = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[0]), dtype=np.complex64)

        for i in tqdm(range(len(f))): # i refers to the freq.
            U,B = self.ica(X[:,i,:])
            epsilon_v[:,i,:], v[:,i,:,:] = self.get_epsilon(U, B)
        
        sim = np.zeros_like(f)
        for i in range(len(f)):
            sim[i] = self.get_sim(epsilon_v, i)

        odr_sim = np.argsort(-sim, kind='heapsort')
        
        y = np.zeros_like(v, dtype=np.complex64)
        epsilon_y = np.zeros_like(epsilon_v)

        n = epsilon_v.shape[0]

        y[:,0,:,:] = v[:,odr_sim[0],:,:]
        epsilon_y[:,0,:] = epsilon_v[:,odr_sim[0],:]

        for k, w_k in enumerate(tqdm(odr_sim)):
            if(k==0): 
                continue

            #create matrix for correlation
            crlat = np.zeros((n,n))
            for a in range(n):
                for b in range(n):
                    for j in range(k-1):
                        w_j = odr_sim[j]
                        crlat[a][b] += np.sum(epsilon_v[b,w_k,:]*epsilon_y[a,w_j,:])
                    
            #complete matching with munkres algorithm                   
            munkres = Munkres()
            indexes = munkres.compute(-crlat)
                
            for i in range(n):
                y[i,w_k,:,:] = v[indexes[i][1],w_k,:,:]

            epsilon_y[:,w_k,:] = self.make_epsilon(y[:,w_k,:,:])

        return y

    def get_epsilon(self, U, B):
        '''
        for specific frequency w.
        @input(U): 2 dimensional complex ndarray. x-axis is channel index, y-axis time segment.
        @input(B): 2 dimensional complex ndarray. x,y-axies are channel indices.
        @output(v): 3 dimensional ndarray. z-axis is channel index j.
        '''

        n, TS  = U.shape
        epsilon_v = np.zeros((n, TS))
        v = np.zeros((n,TS,n), dtype=np.complex64)
        sum_v = np.zeros((n,TS), dtype=np.complex64)
        
        for ts in range(TS):
            v[:,ts,:] = np.dot(inv(B), np.diag(U[:,ts].flatten())).T
            
        epsilon_v = self.make_epsilon(v)

        return epsilon_v, v
        

    def make_epsilon(self, v):
        '''
        This yeilds the epsilon of v from v.
        @param(v): 3 dimensional array whose x,z axis are source n, y is segment times.
        @output(epsilon_v): real value.
        '''
        n, TS, _  = v.shape
        epsilon_v = np.zeros((n, TS))
        sum_v = np.sum(np.abs(v), axis=2)
        
        for ts in range(TS):
            for dts in range(np.maximum(0,ts-self.m_shit), np.minimum(TS, ts+self.m_shit+1)):
                epsilon_v[:,ts]  += 1/(2*self.m_shit) * sum_v[:, dts]

        return epsilon_v

    def epsilon_dot(self, epsilon_v, w1_i, i, w2_i, j):
        '''
        @param(epsilon_v): is 3 dimentional array. z-axis denotes the frequency.
        @param(w1,i): those are set of frequency index and microphone index, which is also the case with (w2,j)
        '''
        return np.sum(epsilon_v[i,w1_i,:] * epsilon_v[j,w2_i,:])

    def epsilon_abs(self, epsilon_v, w_i, i):
        
        return np.sqrt(self.epsilon_dot(epsilon_v, w_i, i, w_i, i)) 

    
    def get_sim(self, epsilon_v , w_i):
        '''
        @param(w): frequency indices
        @output(sim): cross correlation.
        '''
        n = epsilon_v.shape[0]
        sim = .0
        for i in range(n-1):
            for j in range(i,n):
                sim += self.epsilon_dot(epsilon_v, w_i, i, w_i, j)/(self.epsilon_abs(epsilon_v, w_i, i)*self.epsilon_abs(epsilon_v, w_i, j))
        return sim

