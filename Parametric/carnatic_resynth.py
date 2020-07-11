# Script to check cc by reconstructing the audio

# Dependencies

import glob
import pickle
from time import time

from scipy.signal import windows
from scipy.io.wavfile import write
from scipy import interpolate
import numpy as np

import sys
sys.path.append('../Dependencies/')
sys.path.append('../Dependencies/models/')
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
import harmonicModel as HM
import utilFunctions as UF
import stochasticModel as STM
import morphing as m
import func_envs as fe
from essentia.standard import MonoLoader, StartStopSilence, FrameGenerator
import essentia

# Pointer to pickle data dump saved in the preprocessing stage
data = pickle.load(open('./Carnatic_Processing_Dump_B_CHpS.txt', 'rb'))

params = {}
params['fs'] = 44100
Fs = params['fs']
params['W'] = 1024
W = params['W']
params['N'] = 2048
N = params['N']
params['H'] = 256
H = params['H']
params['t'] = -120
params['maxnSines'] = 150
params_ceps = {}
params_ceps['thresh'] = 0.1
params_ceps['num_iters'] = 1000
params_R = {}
params_R['Fr'] = 512
params_R['Hr'] = 256
params_R['ceps_coeffs_residual'] = 50


t = 0
for k in data.keys():
	print (t+1,'/',len(data.keys()))
	t = t + 1
	fHz = data[k]['pitch']


	new_F = np.zeros((data[k]['cc'].shape[0],params['maxnSines']))
	new_M = np.zeros_like(new_F)

	for i in range(fHz.shape[0]):
		hm_arr = np.arange(params['maxnSines']) + 1
		f0 = fHz[i]
		new_F[i,:] = f0*hm_arr 

	for i in range(new_F.shape[0]):

		# Synthesis of Harmonic Component
		fbins = np.linspace(0,params['fs'],params['N'])
		frame = data[k]['cc'][i]
		zp = np.pad(frame,[0,params['N'] - len(frame)],mode = 'constant',constant_values=(0, 0))
		zp = np.concatenate((zp[:params['N']//2],np.flip(zp[1:params['N']//2 + 1])))
		specenv = np.real(np.fft.fft(zp))
		fp = interpolate.interp1d(np.arange(params['N']//2),specenv[:params['N']//2],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		new_M[i,:] = 20*fp((new_F[i,:]/params['fs'])*params['N'])

	# print(data[k]['res'].shape[0])
	for i in range(data[k]['res'].shape[0]):
		# Synthesis of Residual Component
		frame = data[k]['res'][i]
		zp = np.pad(frame,[0,params_R['Fr'] - len(frame)],mode = 'constant',constant_values=(0, 0))
		zp = np.concatenate((zp[:params_R['Fr']//2],np.flip(zp[1:params_R['Fr']//2 + 1])))
		mY = np.real(np.fft.fft(zp))
		# print(mY.shape)
		mY = 20*mY[:params_R['Fr']//2 + 1]
		# print(mY.shape)
		if(i == 0):
			stocEnv = np.array([mY])
		else:
			stocEnv = np.vstack((stocEnv, np.array([mY])))

	harmonic = sineModelSynth(new_F, new_M, np.empty([0,0]), W, H, Fs)
	stochastic_residual = STM.stochasticModelSynth(stocEnv, params_R['Hr'], params_R['Fr'])

	# Add the residual to the harmonic to obtain the audio
	ltw = min(len(harmonic),len(stochastic_residual))

	hps_recon_audio = harmonic[:ltw] + stochastic_residual[:ltw]

	# Save the reconstructed audio
	write(filename = './test/' + str(k) + '_reconH.wav', rate = params['fs'], data = harmonic.astype('float32'))
	write(filename = './test/' + str(k) + '_reconR.wav', rate = params['fs'], data = stochastic_residual.astype('float32'))
	write(filename = './test/' + str(k) + '_recon.wav', rate = params['fs'], data = hps_recon_audio.astype('float32'))