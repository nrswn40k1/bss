import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
from scipy.signal import stft, istft

epsilon = 1e-6


# suppose that the number of sources and microphones are equal.


### Definitions ###

# N : # of channels whose index is n
# I : # of frequency bins whose index is i
# J : # of time frames whose index is j


class ILRMA:

    def __init__(self, x, sample_freq, L=2, max_iter=200, win='hamming', nperseg=256, noverlap=128):
        '''
        @param(win):str, desired window to use.
        @param(nperseg): length of each segment.
        @param(noverlap): number of points to overlap between segments.
        '''
        self.max_iter = max_iter
        self.eta = 2.5 * 10 ** (-2)  # is step size
        self.x = np.array(x)
        self.sample_freq = sample_freq
        self.L = L
        self.win = win
        self.nperseg = nperseg
        self.noverlap = noverlap

    def ilrma(self):
        '''
        X is complex64-type-3-dementional array whose x axis is microphie , y axis is the segment times, z is frequency respectively.
        @output(x_prd): 2 dimensional array whose 1st axis is the source index, 2nd is data of them.
        '''

        f, _, X = stft(self.x, self.sample_freq, self.win, self.nperseg, self.noverlap)
        # X is (channel index, freq index, time segment index)

        y = self.reconstruct(X, self.L)

        _, x_prd = istft(y, self.sample_freq, self.win, self.nperseg, self.noverlap)

        return x_prd

    def reconstruct(self, x, L):
        '''
        This func is the way of permutation.
        @param(f): frequency array.
        @param(X): stft of time series x.
        @output(y):y is 3 dementional array
                   whose 1st axis is source index 2nd axis is frequency index and 3rd is time segment index.
        '''

        W = self.optimize(x, L)
        y = np.zeros_like(x, dtype=np.complex64)
        N, I, J = x.shape
        for i in range(I):
            y[:,i,:] = np.dot(W[:,:,i], x[:,i,:])
        return y

    def optimize(self, x, L):
        N, I, J = x.shape
        W = np.zeros((N,N,I), dtype=np.complex64)
        y = np.zeros_like(x, dtype=np.complex64)
        P = np.zeros_like(x, dtype=float)
        R = np.zeros_like(x, dtype=float)

        """Initialize"""
        for i in range(I):
            W[:,:,i] += np.eye(N)
        T = np.abs(np.random.randn(N,I,L))
        V = np.abs(np.random.randn(N,L,J))

        """Calculate"""
        for i in range(I):
            y[:,i,:] = np.dot(W[:,:,i], x[:,i,:])
        for n in range(N):
            P[n,:,:] += np.abs(y[n,:,:])**2                             # initial power spectorams of estimated sources
            R[n,:,:] += np.dot(T[n,:,:], V[n,:,:])                      # initial model power spectorams

        """Optimization"""
        for _ in tqdm(range(self.max_iter)):
            for n in range(N):
                alpha = np.dot(P[n,:,:]*R[n,:,:]**(-2), V[n,:,:].T)
                alpha = alpha / np.dot(R[n,:,:]**(-1), V[n,:,:].T)
                alpha = np.sqrt(alpha)

                T[n,:,:] = T[n,:,:] * alpha
                T[n,:,:][T[n,:,:] < epsilon] = epsilon                  # update of basis matrix

                R[n,:,:] = np.dot(T[n,:,:], V[n,:,:])                   # new model spectrograms

                beta = np.dot(T[n,:,:].T, P[n,:,:]*R[n,:,:]**(-2))
                beta = beta / np.dot(T[n,:,:].T, R[n,:,:]**(-1))
                beta = np.sqrt(beta)

                V[n,:,:] = V[n,:,:] * beta
                V[n,:,:][V[n,:,:] < epsilon] = epsilon                  # update of activation matrix

                R[n, :, :] = np.dot(T[n, :, :], V[n, :, :])             # new model spectrograms

                for i in range(I):
                    U = np.dot(x[:,i,:]*(R[n,i,:]**(-1)*np.ones((N,1))), np.conjugate(x[:,i,:].T))
                    U = U / J                                           # U is M x M matrix

                    w_ni = inv(np.dot(W[:,:,i], U))[:,n]                # update of demixing filter
                    w_ni = w_ni * np.sqrt(np.dot(np.conjugate(w_ni), np.dot(U, w_ni)))
                                                                        # normalization of demixing filter
                    W[n,:,i] = np.conjugate(w_ni)

            for i in range(I):
                y[:,i,:] = np.dot(W[:,:,i], x[:,i,:])                   # new estimated sources
            for n in range(N):
                P[n,:,:] = np.abs(y[n,:,:])**2                          # new power spectograms of estimated sources
            for n in range(N):
                lam = np.sqrt(np.mean(P[n,:,:]))
                W[n,:,:] = W[n,:,:] / lam
                P[n,:,:] = P[n,:,:] / lam**2
                R[n,:,:] = R[n,:,:] / lam**2
                T[n,:,:] = T[n,:,:] / lam**2

        return W

