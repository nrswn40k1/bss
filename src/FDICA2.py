import numpy as np
from numpy.linalg import inv
from scipy.signal import stft, istft
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
import sys
import warnings

#suppose that the number of sources and microphones are equal.

class ICA:
    '''
    @func(__fai_func_sigmoid): use sigmoid as fai func.
    @func(__fai_func_sign): use sign functio as fai func.
    @func(__fai_func_tanh): use tanh as fai func.
    '''
    
    def __init__(self, num_iter=200):
        self.max_iter = num_iter
        self.eta = 1.0e-5 # is step size
        print('TDICA iteration: {} [times]'.format(self.max_iter))

    def ica(self, x):
        x = np.array(x)
        w = self.__optimize(x)
        y = np.dot(w, x)
        return y, w

    def __fai_func_sigmoid(self, y): 
        return 1.0/(1.0+np.exp(-y.real)) + 1.0j*1.0/(1.0+np.exp(-y.imag))

    def __fai_func_tanh(self,y):
        return np.tanh(100.0 * y)

    def _alpha(self, y):
        '''
        You can change the __fai_func_xxxxx from 3 different function above.
        '''
        return  np.dot(self.__fai_func_sigmoid(y), y.T.conjugate())    

    def _optimize(self, x):
        r,c = x.shape
        w = np.zeros((r,r), dtype=np.complex64)
        w += np.diag(np.ones(r))
        
        
        for _ in range(self.max_iter):
            y = np.dot(w, x)
            alpha = self.__alpha(y)
            
            alpha = alpha/c
            w += self.eta * np.dot((np.diag(np.diag(alpha)) - alpha),  w)
        return w


class FDICA(ICA):
	
	def __init__(self, x, sample_freq, mic_array, num_iter=200, win='boxcar', nperseg=256, noverlap=126):
		'''
		@param(mic_array):the distance of microphone from the origin like [d0, d1, d2,...]. Note that the unit is m not cm, mm.
		@param(n_iter): the times of iteration of TDICA optmization.
		@param(win):str, desired window to use.
		@param(nperseg): length of each segment.
		@param(noverlap): number of points to overlap between segments.
		* (nperseg, noverlap) = (1024, 512) 
		'''
		print('-----------------------------------------')
		super().__init__(num_iter=num_iter)
		self.num_theta = 1000
		self.c = 340.29
		self.num_speaker = x.shape[0]
		self.x = np.array(x)
		self.d = mic_array
		self.sample_freq = sample_freq
		self.win = win
		self.nperseg = nperseg
		self.noverlap = noverlap

		if x.shape[0] != mic_array.shape[0]:
			print('# of the samples and # of the microphones are inconsistent.')
			sys.exit()

		print('The sample frequency: {} [/sec]'.format(sample_freq))
		print('The length of each segment: {}'.format(nperseg))
		print('The number of points to overlap between segments: {}'.format(noverlap))

	def fdica(self):
		'''
		X is complex64-type-3-dementional array whose x axis is microphie , y axis is the segment times, z is frequency respectively.
		@output(x_prd): 3 dimensional array whose 1st axis is the source index, 2nd is the microphon index, third is data of them.
		'''
		start = time.time()
		print('Now... short time discrete fourier transformation')

		f,_,X = stft(self.x, self.sample_freq, self.win, self.nperseg, self.noverlap)

		# X is (channel index, freq index, time segment idex)

		y = self.__separate(f,X)

		print('Now... inverted short time discrete fourier transformation')

		_,x_prd = istft(y, self.sample_freq, self.win, self.nperseg, self.noverlap)

		deltatime = time.time()-start
		print('FDICA took {} [sec] to finish'.format(deltatime))
		print('-----------------------------------------')
		return x_prd
	
	def __separate(self, f, X):
		'''
		the whole process of fdica except isft.
		'''
		y = np.zeros((X.shape[0], X.shape[1], X.shape[2]), dtype=np.complex64)
		print('Now... separation in each {} frequency.'.format(len(f)))
		
		for i in tqdm(range(len(f))):
			"""(step 1) initialization of w"""
			w = np.ones((X.shape[0], X.shape[0]), dtype=np.complex64)
			#w = np.eye(X.shape[0], dtype=np.complex64)
			x = X[:,i,:]
			thetas = np.zeros(self.num_speaker)
			f[i] = float(f[i])

			for _ in range(self.max_iter):
				W_ica = self.__ica_optimize(w, x)
				try:
					thetas = self.__doa_estimation(W_ica, f[i])
					W_bf = self.__beamforming(thetas, f[i])
					w = self.__diversity_with_cost_func(W_ica, W_bf, x)
				except:
					w = W_ica
			
			y[:,i,:] = self.__ordering_and_scaling(w, x, thetas, f[i])
		
		return y

	def __ica_optimize(self, w, x):
		'''
		(step 2) 1-time ICA iteration.
		@input(x):2-dimensional data in one frequency bin.
		''' 
		r,c = x.shape
        
		y = np.dot(w, x)
		alpha = super()._alpha(y)/c
		w += self.eta * np.dot((np.diag(np.diag(alpha)) - alpha),  w)

		return w
	
	def __doa_estimation(self, w, freq):
		'''
		(step 3) DOA estimation in one frequency bin.
		''' 
		e = np.zeros((self.d.shape[0], self.num_theta), dtype=np.complex64)
		self.d = self.d.reshape((self.d.shape[0],1))
		e = np.exp(1.0j*2.0*np.pi*freq/self.c*np.dot(self.d, np.sin(np.linspace(-np.pi/2, np.pi/2, self.num_theta)).reshape((1,self.num_theta))))
		F = np.abs(np.dot(w, e))
		F_diff = np.diff(F, axis=-1)
		first_condi = np.where(F_diff[:,:-1] <= 0, 1, 0) #first condition satisfied = 1 
		second_condi = np.where(F_diff[:,1:] > 0, 1, 0) # second condition satisfied = 1
		satisfied = first_condi*second_condi
		candidate_theta = np.linspace(-np.pi/2, np.pi/2, self.num_theta)[(np.where(satisfied==1)[1] + 1)]

		"""LIoyd clustering algorithm"""
		try:
			with warnings.catch_warnings():
				warnings.simplefilter("error")
				km = KMeans(n_clusters=self.num_speaker)
				km.fit(candidate_theta.reshape(-1,1))
		except Exception as e:
			raise
		
		return  km.cluster_centers_
		
	def __beamforming(self, thetas, freq):
		'''
		(step 4) Beamforming in one frequency bin.
		@output(): W_bf
		'''
		try:
			e_hat = np.zeros((self.d.shape[0], self.num_speaker), dtype=np.complex64)
			self.d = self.d.reshape((self.d.shape[0],1))
			e_hat = np.exp(1j*2.0*np.pi*freq/self.c*np.dot(self.d, np.sin(thetas).reshape((1,self.num_speaker))))
			return inv(e_hat)
		except Exception as e:
			raise

	def __diversity_with_cost_func(self, W_ica, W_bf, x):
		'''
		(step 5) Diversity using cost function in one frequency bin.
		@input(x): frequency-domain data of a certain freq.
		'''
		if self.__cost_func(W_ica, x) <= self.__cost_func(W_bf, x):
			print(W_ica)
			return W_ica
		else:
			return W_bf

	def __cost_func(self, W, x):
		'''
		internal function used in __diversity_with_cost_func.
		'''
		y = np.dot(W,x)
		corr_y = np.abs(np.corrcoef(y))
		return np.sum(corr_y - np.diag(np.diag(corr_y))) 

	def __ordering_and_scaling(self, W, x, thetas, freq):
		'''
		(step 6) Ordering and scaling.
		'''
		e_hat = np.zeros((self.d.shape[0], self.num_speaker), dtype=np.complex64)
		self.d = self.d.reshape((self.d.shape[0],1))
		e_hat = np.exp(1j*2.0*np.pi*freq/self.c*np.dot(self.d, np.sin(thetas).reshape((1,self.num_speaker))))
		F = np.abs(np.dot(W, e_hat))
		threshold = np.min(np.max(F, axis=1))
		F = np.where(F >= threshold, 1/F, 0).T
		return np.dot(F, np.dot(W, x))