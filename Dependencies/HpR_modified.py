"""
The original HpR code subtracts the Harmonic model from the original audio to compute the residual. In our case, we compute the residual
by subtracting the TAE based SF representation (on top of the Harmonic model) from the original audio.
This is done to obtain a identity transformation of the audio (i.e. obtain a complete representation of the original audio)
"""


# Dependencies
import glob
import pickle
from time import time

from scipy.signal import windows
from scipy import interpolate
import numpy as np

import sys
# sys.path.append('/home/krishna/Desktop/IITB/DDP/Main/ddp-genmodels/experiments/models/')
sys.path.append('./')
sys.path.append('./models/')
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
import harmonicModel as HM
import utilFunctions as UF
import stochasticModel as STM
# sys.path.append('/home/krishna/Desktop/IITB/DDP/Main/Sound-Morphing-Guided-by-Perceptually-Motivated-Features/Experiments/')
# sys.path.append('/home/krishna/Desktop/IITB/DDP/Main/Sound-Morphing-Guided-by-Perceptually-Motivated-Features/Experiments/Krishna')
# sys.path.append('/home/krishna/Desktop/IITB/DDP/Main/Sound-Morphing-Guided-by-Perceptually-Motivated-Features/Experiments/Krishna/SMS_Morphing/')
import morphing as m
import func_envs as fe
from essentia.standard import MonoLoader, StartStopSilence, FrameGenerator, PitchYinProbabilistic, PitchYin
import essentia

# Modified Harmonic Model dependencies
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft
import math
import dftModel as DFT
import sineModel as SM


def HpR_modified(audio_inp, params, params_ceps,f0):
	"""
	Computes the Harmonic model and the 'modified' residual.
	Input
	-----
	audio_inp : np.array
		Numpy array containing the audio signal, in the time domain 
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : Sampling rate of the audio
			- W : Window size(number of frames)
			- N : FFT size(multiple of 2)
			- H : Hop size
			- t : Threshold for sinusoidal detection in dB
			- maxnSines : Number of sinusoids to detect
	params_ceps : dict
		Parameter dictionary for the TAE algorithm
			- thersh : dB threshold for the TAE
			- ceps_coeffs : Number of kept cepstral coefficients
			- num_iters : Upper limit on number of iterations to run the TAE loop
	f0 : float
		The fundamental frequency of the audio
	
	Returns
	-------
	harmonic : The harmonic component of the audio
	residual : The residual component of the audio
	hpr : Reconstructed (Harmonic + Residual)
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']
	w = windows.hann(W)

	F, M, P = HM.harmonicModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH=  maxnSines,minSineDur = 0.02, minf0 = f0 - 10, maxf0 = f0 + 10, f0et = 5, harmDevSlope = 0.01)

	# Define the new magnitude array based on the modified coefficients sampled from the TAE
	new_M = np.zeros_like(M)
	new_F = np.zeros_like(F)
	new_P = np.zeros_like(P)

	for i in range(new_F.shape[1]):
		new_F[:,i] = (i+1)*f0
	# new_F = F
	# new_M = M
	# Compute the TAE over the Harmonic model

	for i in range(F.shape[0]):
		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f = interpolate.interp1d((F[i,:]/fs)*N,M[i,:]/20,kind = 'linear',fill_value = '-6', bounds_error=False)
		# Frequency bins
		# fbins = np.linspace(0,fs/2,N//2)
		fbins = np.arange(N)
		finp = f(fbins)
		zp = np.concatenate((finp[0:N//2],np.array([0]),np.flip(finp[1:N//2])))
		# print(zp.shape)
		specenv,_,_ = fe.calc_true_envelope_spectral(zp,N,thresh,ceps_coeffs,num_iters)
		fp = interpolate.interp1d(np.arange(N//2),specenv[:N//2],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		new_M[i,:] = 20*fp((new_F[i,:]/params['fs'])*params['N'])


	# Compute the residual by subtracting the above from the original audio
	Ns = N
	residual = UF.sineSubtraction(audio_inp,Ns,H,new_F,new_M,P,fs)

	# Obtain the audio from the Harmonic Model
	harmonic = sineModelSynth(new_F, new_M, P, W, H, fs)

	# Add the residual to the harmonic to obtain the audio
	ltw = min(len(harmonic),len(residual))

	hpr = harmonic[:ltw] + residual[:ltw]

	return harmonic,residual,hpr


def HpS_modified(audio_inp, params, params_ceps,f0):
	"""
	Computes the Harmonic model and the stochastic version of the 'modified' residual.
	Input
	-----
	audio_inp : np.array
		Numpy array containing the audio signal, in the time domain 
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : Sampling rate of the audio
			- W : Window size(number of frames)
			- N : FFT size(multiple of 2)
			- H : Hop size
			- t : Threshold for sinusoidal detection in dB
			- maxnSines : Number of sinusoids to detect
	params_ceps : dict
		Parameter dictionary for the TAE algorithm
			- thresh : dB threshold for the TAE
			- ceps_coeffs : Number of kept cepstral coefficients
			- num_iters : Upper limit on number of iterations to run the TAE loop
			- ceps_coeffs_residual : Number of kept cepstral coefficients for the residual envelope
	f0 : float
		The fundamental frequency of the audio
	
	Returns
	-------
	harmonic : The harmonic component of the audio
	stochastic_residual : The stochastic residual component of the audio
	hps : Reconstructed (Harmonic + Stochastic Residual)
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']
	w = windows.hann(W)

	F, M, P = HM.harmonicModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH=  maxnSines,minSineDur = 0.02, minf0 = f0 - 10, maxf0 = f0 + 10, f0et = 5, harmDevSlope = 0.01)

	# Define the new magnitude array based on the modified coefficients sampled from the TAE
	new_M = np.zeros_like(M)
	new_F = np.zeros_like(F)
	new_P = np.zeros_like(P)

	for i in range(new_F.shape[1]):
		new_F[:,i] = (i+1)*f0
	# new_F = F
	# new_M = M
	# Compute the TAE over the Harmonic model

	for i in range(F.shape[0]):
		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f = interpolate.interp1d((F[i,:]/fs)*N,M[i,:]/20,kind = 'linear',fill_value = '-6', bounds_error=False)
		# Frequency bins
		# fbins = np.linspace(0,fs/2,N//2)
		fbins = np.arange(N)
		finp = f(fbins)
		zp = np.concatenate((finp[0:N//2],np.array([0]),np.flip(finp[1:N//2])))
		# print(zp.shape)
		specenv,_,_ = fe.calc_true_envelope_spectral(zp,N,thresh,ceps_coeffs,num_iters)
		fp = interpolate.interp1d(np.arange(N//2),specenv[:N//2],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		new_M[i,:] = 20*fp((new_F[i,:]/params['fs'])*params['N'])

	# Obtain the audio from the Harmonic Model
	harmonic = sineModelSynth(F, M, np.empty([0,0]), W, H, fs)

	# Obtaining the residual and stochastic approximation
	# Compute the residual by subtracting the above from the original audio
	Ns = N
	residual = UF.sineSubtraction(audio_inp,Ns,H,new_F,new_M,P,fs)
	residual_og = UF.sineSubtraction(audio_inp,Ns,H,F,M,P,fs)

	# Compute the framewise TAE for the 'stochastic residual' as an approximation of the residual envelope
	cou = 0
	ceps_coeffs_residual = params_ceps['ceps_coeffs_residual']
	for frame in FrameGenerator(essentia.array(residual), frameSize = W, hopSize = H):
		mY,_,_,_ = fe.calc_true_envelope(frame*w,512,thresh,ceps_coeffs_residual,1)
		mY = 20*mY[:257]
		if(cou == 0):
			stocEnv = np.array([mY])
		else:
			stocEnv = np.vstack((stocEnv, np.array([mY])))
		cou = cou + 1
	# print(stocEnv.shape)

	stocEnv_og = STM.stochasticModelAnal(residual_og, H, 512, 1)

	# print(stocEnv.shape)
	# Reconstruct the residual with the above 'stochastic envelope'
	stochastic_residual = STM.stochasticModelSynth(stocEnv, H, 512)

	# stochastic_residual = residual

	# Add the residual to the harmonic to obtain the audio
	ltw = min(len(harmonic),len(stochastic_residual))

	hps = harmonic[:ltw] + stochastic_residual[:ltw]

	return harmonic,stochastic_residual,hps,stocEnv,stocEnv_og


def harmonicModel_pyin(x, fs, w, N, H, t, nH, harmDevSlope=0.01, minSineDur=.02):
	"""
	Same as above, except, uses the pYIN pitch contour as the pitch estimate instead of the TwM normally used.
	"""
	# Define the P-YIN initializer in Essentia
	pyin = PitchYinProbabilistic(frameSize = w.shape[0],hopSize = H,lowRMSThreshold = 0.0001,preciseTime = True)

	if (minSineDur <0):                                     # raise exception if minSineDur is smaller than 0
		raise ValueError("Minimum duration of sine tracks smaller than 0")
		
	hN = N//2                                               # size of positive spectrum
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample

	# Instead of obtaining the framewise pitch using the TwM algorithm, we will first find the pitch using P-YIN, and then use that as the framewise pitch
	f0t_pyin,vp = pyin(essentia.array(x))
	f0t_pyin = np.asarray(f0t_pyin)

	pin = hM1                                               # init sound pointer in middle of anal window          
	pend = x.size - hM1                                     # last sample to start a frame
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	w = w / sum(w)                                          # normalize analysis window
	hfreqp = []                                             # initialize harmonic frequencies of previous frame
	f0t = 0                                                 # initialize f0 track
	f0stable = 0                                            # initialize f0 stable

	pi = 0 # pitch pointer/iterator variable
	while pin<=pend:           
		x1 = x[pin-hM1:pin+hM2]                               # select frame
		mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft            
		ploc = UF.peakDetection(mX, t)                        # detect peak locations   
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
		ipfreq = fs * iploc/N                                 # convert locations to Hz

		# Here, we replace the TwM pitch with the P-YIN pitch estimate
		f0t = f0t_pyin[pi]
		pi = pi + 1		# Advance the pitch pointer
		# f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
		if ((f0stable==0)&(f0t>0)) \
				or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
			f0stable = f0t                                      # consider a stable f0 if it is close to the previous one
		else:
			f0stable = 0
		hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, harmDevSlope) # find harmonics
		hfreqp = hfreq
		if pin == hM1:                                        # first frame
			xhfreq = np.array([hfreq])
			xhmag = np.array([hmag])
			xhphase = np.array([hphase])
		else:                                                 # next frames
			xhfreq = np.vstack((xhfreq,np.array([hfreq])))
			xhmag = np.vstack((xhmag, np.array([hmag])))
			xhphase = np.vstack((xhphase, np.array([hphase])))
		pin += H                                              # advance sound pointer
	xhfreq = SM.cleaningSineTracks(xhfreq, round(fs*minSineDur/H))     # delete tracks shorter than minSineDur
	return xhfreq, xhmag, xhphase



