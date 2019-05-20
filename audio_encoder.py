import numpy as np
from scipy.io import wavfile
from keras import Sequential
from keras.layers import Dense, Conv1D, Reshape, Flatten
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from scipy import signal
from multiprocessing_generator import ParallelGenerator

class audioEncoder:

# Control Parameters
	windowSize = 257*100
	encodedDim = 60 	#Error reduction dropped off around 18, with window 512

	model_select = 1

	plot_EN 		=	[False,True][0]
	train_EN 		=	[False,True][1]
	LPF_EN			=	[False,True][0]
	stft_EN			=	[False,True][1]
	search_len 		=	[False,True][0]
	bypass_phase 	=	[False,True][1]

	wd = '/media/josh/9d2726f1-95f3-4f4f-99dc-3c6859407be0/josh/Documents/audio/'
	train_fn 	= 	wd+'bowed_bass.wav'
	test_fn 	= 	wd+'bass_test.wav'

	num_train_seconds = 3
	num_test_seconds = 8
	Fs = 44100
	train_step = 32

# Shared Class state
	debug_toolA = []
	debug_toolB = []
	wav = []
	train = []
	test = []
	model = ''
	og = []
	re = []
	error = []
	errors = []
	phase = []

# Init
	def __init__(self):
		# if self.stft_EN:
		# 	self.windowSize += 2
		self.main()

# Main
	def main(self):
		if self.search_len:
			self.load_audio('train')
			self.num_train_seconds = 10
			self.num_test_seconds = 10
			self.hidden_layer_search()
		else:
			self.init_model()
			if self.train_EN:
				self.load_audio('train')
				self.train_model()
			else:
				self.load_model()
			self.load_audio('test')
			self.test_model()
			self.display_results()
			self.write_audio()

# Load wav
	def load_audio(self,train_test):
		# String bass
		if train_test == 'train':
			self.Fs,self.wav = wavfile.read(self.train_fn)
		else:
			self.Fs,self.wav = wavfile.read(self.test_fn)

	# cut to mono for simplicity

# window the wav
	def windowGen(self,x,length,step,phase=[]):
		l = x.shape[0]
		for i in range(l-length+1):
			if not i%step:
				idx = [ix for ix in range(i,i+length)]
				if len(phase):
					if not len(self.phase):
						self.phase = np.take(phase,idx,axis=0)
					else:
						self.phase = np.concatenate([self.phase,np.take(phase,idx,axis=0)])
				yield(np.abs(np.take(x,idx,axis=0)))

# Generator Zipper 1,1 -> (1,1)
	def xygen(self,xg,yg):
		for x,y in zip(xg,yg):
			# yield(x[None,:],y[None,:])
			yield(x,y)

# Normalize gen
	def windowNormGen(self,xg):
		for x in xg:
			bias = x.min()
			x = x-bias
			amplitude = x.max()
			x = x/amplitude
			yield(x)

# Normalize gen
	def windowNormGen_reconstructable(self,xg):
		for x in xg:
			bias = x.min()
			x = x-bias
			amplitude = x.max()
			x = x/amplitude
			yield([x,bias,amplitude])

# Accumulator gen for speed
	def accumulatorGen(self,xg,l=1000):
		ctr = -1
		for x in xg:

			if ctr == -1:
				ctr = 0
				shape = [l]
				for v in x.shape:
					shape.append(v)
				a = np.zeros(shape)

			a[ctr] = x

			ctr += 1

			if ctr == l:
				ctr = 0
				yield(a)

		yield(a)

# LPF
	def lowpass(self,x,Fs):
		sos = signal.butter(N=12,Wn=20000/Fs,btype='lowpass',output='sos')
		# b,a = signal.butter(N=12,Wn=20000/Fs,btype='lowpass')
		# w,gd = signal.group_delay((b,a))
		filtered = signal.sosfilt(sos,x)
		return(filtered)


	# for encodedDim in range(1,encodedDim_max+1):

# init ae
	def init_model(self):
		if self.model_select == 1:
			self.init_M1()
		elif self.model_select == 2:
			self.init_M2()

# AEC M1
	def init_M1(self):
		model = Sequential()

		model.add(Dense(self.encodedDim,activation='relu',input_dim=(self.windowSize)))

		model.add(Dense(self.windowSize,activation='sigmoid'))

		model.compile(loss='mse', optimizer='adam')

		self.model = model

		self.model.summary()

# AEC M2
	def init_M2(self):
		model = Sequential()

		model.add(Dense(self.encodedDim,activation='relu',input_dim=(self.windowSize)))

		model.add(Dense(self.windowSize,activation='sigmoid'))

		model.add(Reshape((self.windowSize,1)))

		model.add(Conv1D(filters=1,kernel_size=64,padding='same'))

		model.add(Flatten())

		model.compile(loss='mse', optimizer='adam')

		self.model = model

		self.model.summary()

# AEC Test Hidden
	def init_aec_test(self,size):
		model = Sequential()

		model.add(Dense(size,activation='relu',input_dim=(self.windowSize)))

		model.add(Dense(self.windowSize,activation='sigmoid'))

		model.compile(loss='mse', optimizer='adam')

		self.model = model

		self.model.summary()

# Post MLP Conv Filter

# Get Data
	def get_data(self,train_test):

		if train_test == 'train':
			# Cut to mono
			w = self.wav[:self.Fs*self.num_train_seconds,0]
		else:
			# test on self
			w = self.wav[self.Fs*self.num_train_seconds:self.Fs*self.num_train_seconds+self.Fs*self.num_test_seconds,0]
			# w = self.wav[:self.Fs*self.num_train_seconds,0]

		if self.stft_EN:
			_,_,w = signal.stft(w,fs=self.Fs,nperseg=512)
			a = np.abs(w)
			p = np.angle(w)
			if self.bypass_phase:
				w = a
				w = a.flatten()
				p = p.flatten()
				if train_test == 'test':
					return([w,p])
			else:
				w = np.array([a,p]).transpose().flatten()

		return(w)

# Train Model
	def train_model(self):

		w = self.get_data('train')

		# train generator ae
		wgen1 = self.windowGen(w,self.windowSize,step=int(self.train_step))
		wgen2 = self.windowGen(w,self.windowSize,step=int(self.train_step))

		wn1 = self.windowNormGen(wgen1)
		wn2 = self.windowNormGen(wgen2)

		wn1a = self.accumulatorGen(wn1)
		wn2a = self.accumulatorGen(wn2)

		# xyg = self.xygen(wn1a,wn2a)

		# for x,y in xyg:
		# 	self.model.fit(x,y,epochs=1)

		with ParallelGenerator(
			self.xygen(wn1a,wn2a),
			max_lookahead=200) as xyg:
			for x,y in xyg:
				self.model.fit(x,y,epochs=1)

		self.save_model()
		print('Model Saved')

# Load Model
	def load_model(self):
		self.model.load_weights('aec_w.h5')

# Save Model
	def save_model(self):
		self.model.save_weights('aec_w'+'.h5')

# Test Model
	def test_model(self):

		if self.stft_EN and self.bypass_phase:
			w,p = self.get_data('test')
			wgen1 = self.windowGen(w,self.windowSize,step=self.windowSize,phase=p)
		else:
			w = self.get_data('test')
			wgen1 = self.windowGen(w,self.windowSize,step=self.windowSize)

		wn1 = self.windowNormGen(wgen1)

		og = []
		re = []

		for x in wn1:
			og.append(x)
			re.append(self.model.predict(x[None,:]))

		re = np.concatenate(np.array(re)[:,0,:],axis=0)
		og = np.concatenate(np.array(og),axis=0)
		
		wgen1 = self.windowGen(w,self.windowSize,step=self.windowSize)
		(gains,biass) = self.get_window_gains(wgen1)
		reg = self.windowGen(re,self.windowSize,step=self.windowSize)
		ogg = self.windowGen(og,self.windowSize,step=self.windowSize)

		# Rescale windows to original gain
		re = self.deNormalize(reg,gains,biass)
		og = self.deNormalize(ogg,gains,biass)

		# Align Windows
		# reg = self.windowGen(re,self.windowSize,step=self.windowSize)
		# re = self.align_windows(reg)



		if self.stft_EN:
			if not self.bypass_phase:
				x = re.reshape((int(re.shape[0]/2),2))
				cx = x[:,0]*np.exp(x[:,1]*(0+1j))
			else:
				cx = re*np.exp(self.phase*(0+1j))
			cx = cx.reshape((257,-1))
			td = signal.istft(cx,fs=self.Fs)[1]
			re = td

			if not self.bypass_phase:
				x = og.reshape((int(og.shape[0]/2),2))
				cx = x[:,0]*np.exp(x[:,1]*(0+1j))
			else:
				cx = og*np.exp(self.phase*(0+1j))
			cx = cx.reshape((257,-1))
			td = signal.istft(cx,fs=self.Fs)[1]
			og = td


		if self.LPF_EN:
			re = self.lowpass(re,self.Fs)
		
		error = abs(re-og)

		self.errors.append(error.mean())

		self.error = error
		self.og = og 
		self.re = re

# Display Results
	def display_results(self):	

		og = self.og 
		re = self.re 
		Fs = self.Fs 
		error = self.error
		print(error.mean())

		if self.plot_EN:
			plt.figure('ogre')
			ogre = np.array([og,re,error])
			plt.imshow(ogre,aspect='auto')

			plt.figure('re')
			# if self.stft_EN:
			# 	plt.plot(re,aspect='auto')
			# else:
			Pxx_re, freqs, bins, im = specgram(re,NFFT=64,noverlap=63,Fs=Fs)

			plt.figure('og')
			# if self.stft_EN:
			# 	plt.plot(og,aspec='auto')
			# else:
			Pxx_og, freqs, bins, im = specgram(og,NFFT=64,noverlap=63,Fs=Fs)

			plt.figure('norm error')
			Pxx_diff = abs(Pxx_og-Pxx_re)
			for r in range(Pxx_diff.shape[0]):
				Pxx_diff[r,:] = Pxx_diff[r,:]-Pxx_diff[r,:].min()
				Pxx_diff[r,:] = Pxx_diff[r,:]/Pxx_diff[r,:].max()

			plt.imshow(np.flip(Pxx_diff),aspect='auto')

			plt.show()

	# test on test

# Write Audio
	def write_audio(self):
		if abs(abs(self.re.max()))>1:
			self.re = self.re/abs(self.re).max()
		if abs(abs(self.og.max()))>1:
			self.og = self.og/abs(self.og).max()
		wavfile.write(open(self.wd+'re.wav','wb'),data=self.re,rate=self.Fs)
		wavfile.write(open(self.wd+'og.wav','wb'),data=self.og,rate=self.Fs)

# Window normalization coefficient ID
	def get_window_gains(self,xg):
		gains = []
		biass = []
		for x in xg:
			bias = x.min()
			gain = (x-x.min()).max()
			biass.append(bias)
			gains.append(gain)

		return([gains,biass])

# Window de-normalization
	def deNormalize(self,xg,gains,biass):
		xout = []
		xs = [x for x in xg]
		for x,gain,bias in zip(xs,gains,biass):
			xo = x*gain
			xo = xo+bias
			xout.append(xo)
		return(np.concatenate(np.array(xout),axis=0))

# Linear window mismatch Correction
	def align_windows(self,xg):
		xp = xg.__next__()
		xout= []
		for x in xg:
			low = 0
			high = xp[-1] - x[0]
			xp = xp-np.linspace(low,high,len(xp))
			xout.append(xp)
			xp = x
		xout.append(xp)
		return(np.concatenate(np.array(xout),axis=0))

# Hidden Parameter Search
	def hidden_layer_search(self):
		# for n in range(1,self.windowSize):
		for n in range(1,128):		
			self.init_aec_test(n)
			self.train_model()
			self.test_model()
			print(self.errors[-1])

		plt.plot(self.errors)
		plt.show()

if __name__ == '__main__':
	aec = audioEncoder()