import numpy as np
from numpy.linalg import inv
from scipy.fftpack import fft


#suppose that the number of sources and microphones are equal.

class SFFT:
    '''
    This is for windowed fourier transformation. 
    '''

    def __init__(self, sample_freq=8000, frame_length=32, frame_shift=16 ):
        '''
        @param(frame_length, frame_shift): The unit of them are ms.
        @param(win):  window function, default 
        '''

        self.sample_freq = sample_freq
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.frame_size = 8000 * (frame_length/1000)
        self.step = 8000 * (frame_shift/1000)
        self.win = np.hamming(self.frame_size)


    def stft(self, x):
        '''
        @input(x): x is supposed to be one-dimensional vector.
        '''

        sample_size = self.x.shape
        m = int(np.ceil(float(sample_size-self.frame_size+self.step)/self.step))
        new_x = np.zeros(self.frame_size+(m-1)*self.step, dtype = np.float64)
        new_x[:l] = x

        X = np.zeros([m, self.frame_size], dtype=np.complex64)
        for i in range(m):
            X[i,:] = fft(new_x[step*i:step*i+self.frame_size] * self.win)
        
        return X
        
    
    def istft(self, x):
        '''
        @input(x): x is supposed to be matrix.
        '''
        

class ICA:
    
    self.max_iter = 500   
    self.eta = 1.0 * 10**-4 # is step size

    def __init__(self,x):
        self.x = np.matrix(x);

    def ica(self, x):
        w = __optimize(x)
        y = np.dot(w,x)
        return y, w

    def __fai_func(self, y):
        return 1/(1+np.exp(-y.real)) + 1j*1/(1+np.exp(-y.imag))

    def __alpha(self, y):
        return  np.dot(self.__fai_func(y), y.H)    

    def __optimize(self, x):
        r,c = x.shape
        w = np.zeros((r,r), dtype=np.complex64)
        
        for _ in range(max_iter):
            y = pre_w * x
            alpha = np.zero((r,r), dtype=np.complex64)

            for i in range(c):
                alpha += self.__alpha(y[:,i])
            
            alpha = alpha/c
            w = self.eta * (np.diag(np.diag(alpha)) - alpha) * inv(w.H)

        return w

    
class FDICA:
    def __init__(self, x):
        self.x = np.matrix(x)

    def FDICA(self):
        
        
        return 

    def 


    
