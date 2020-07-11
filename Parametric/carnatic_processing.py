import glob
import pickle
from time import time

from scipy.signal import windows, resample
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
from essentia.standard import MonoLoader, StartStopSilence, FrameGenerator, PitchYinProbabilistic, PitchYin
import essentia
import HpR_modified as HPRM


# Unlike the ICASSP data processing, here we use the pitch from PYIN (HpR also uses this same pitch estimate!)

def cc_calc(audio_inp, params, params_ceps,f0):

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

	# Cepstral Coefficients Calculation
	CC_harm = np.zeros((F.shape[0], N))
	CC_res = np.zeros((F.shape[0], 512))	

	for i in range(new_F.shape[1]):
		new_F[:,i] = (i+1)*f0

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
		CC_harm[i,:] = np.real(np.fft.ifft(specenv))
		fp = interpolate.interp1d(np.arange(N//2),specenv[:N//2],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		new_M[i,:] = 20*fp((new_F[i,:]/params['fs'])*params['N'])

	# Obtain the audio from the Harmonic Model
	harmonic = sineModelSynth(F, M, np.empty([0,0]), W, H, fs)

	# Obtaining the residual and stochastic approximation
	# Compute the residual by subtracting the above from the original audio
	Ns = N
	residual = UF.sineSubtraction(audio_inp,Ns,H,new_F,new_M,P,fs)

	# Compute the framewise TAE for the 'stochastic residual' as an approximation of the residual envelope
	cou = 0
	ceps_coeffs_residual = params_ceps['ceps_coeffs_residual']
	for frame in FrameGenerator(essentia.array(residual), frameSize = W, hopSize = H):
		if(cou >= F.shape[0]):
			break
		mY,_,_,_ = fe.calc_true_envelope(frame*w,512,thresh,ceps_coeffs_residual,1)
		CC_res[cou,:] = np.real(np.fft.ifft(mY))
		cou = cou + 1
	return CC_harm,CC_res


def cc_calc_varP(audio_inp, params, params_ceps, params_R):
	"""
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	# ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']
	w = windows.hann(W)

	# Parameters for the residual cc calculation
	Fr = params_R['Fr']
	Hr = params_R['Hr']
	ceps_coeffs_residual = params_R['ceps_coeffs_residual']

	# F, M, P = HPRM.harmonicModel_pyin(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH=  maxnSines, minSineDur = 0.02, harmDevSlope = 0.1)
	F, M, P = HM.harmonicModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH=  maxnSines,minSineDur = 0.02, minf0 = 140, maxf0 = 1300, f0et = 5, harmDevSlope = 0.01)

	# Cepstral Coefficients Calculation
	CC_harm = np.zeros((F.shape[0], N))
	# CC_res = np.zeros((F.shape[0], Fr))	

	# Extract the pitch from returned F
	pitch_pyin = F[:,0]
	most_likely_pitch = pitch_pyin[pitch_pyin.shape[0]//2]
	pitch_pyin = most_likely_pitch*np.ones(pitch_pyin.shape)

	# Define the new magnitude array based on the modified coefficients sampled from the TAE
	new_M = np.zeros_like(M)
	new_F = np.zeros_like(F)
	new_P = np.zeros_like(P)

	for i in range(pitch_pyin.shape[0]):
		hm_arr = np.arange(maxnSines) + 1
		f0 = pitch_pyin[i]
		new_F[i,:] = f0*hm_arr 

	# Compute the TAE over the Harmonic model

	for i in range(F.shape[0]):
		f0 = pitch_pyin[i]
		if(f0 != 0):
			ceps_coeffs = (int)(1.9*(fs/(2*f0)))
			# ceps_coeffs = N
		else:
			ceps_coeffs = N

		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f = interpolate.interp1d((F[i,:]/fs)*N,M[i,:]/20,kind = 'linear',fill_value = '-6', bounds_error=False)
		# Frequency bins
		# fbins = np.linspace(0,fs/2,N//2)
		fbins = np.arange(N)
		finp = f(fbins)
		zp = np.concatenate((finp[0:N//2],np.array([0]),np.flip(finp[1:N//2])))
		# print(zp.shape)
		specenv,_,_ = fe.calc_true_envelope_spectral(zp,N,thresh,ceps_coeffs,num_iters)
		CC_harm[i,:] = np.real(np.fft.ifft(specenv))
		fp = interpolate.interp1d(np.arange(N//2),specenv[:N//2],kind = 'linear',fill_value = '-6', bounds_error=False)
		new_M[i,:] = 20*fp((new_F[i,:]/params['fs'])*params['N'])


	# Compute the residual by subtracting the above from the original audio
	# Ns = N
	residual = UF.sineSubtraction(audio_inp,N,H,new_F,new_M,P,fs)

	# Compute the framewise TAE for the 'stochastic residual' as an approximation of the residual envelope
	cou = 0
	for frame in FrameGenerator(essentia.array(residual), frameSize = Fr, hopSize = Hr):
		f0 = pitch_pyin[i]
		if(f0 != 0):
			ceps_coeffs_residual = (int)(1.9*(fs/(2*f0)))
		else:
			ceps_coeffs_residual = Fr
		mY,_,_,_ = fe.calc_true_envelope(frame*windows.hann(Fr),Fr,thresh,ceps_coeffs_residual,num_iters)
		# CC_res[cou,:] = np.real(np.fft.ifft(mY))
		# mY = 20*mY[:Fr//2 + 1]
		if(cou == 0):
			CC_res = np.array([np.real(np.fft.ifft(mY))])
		else:
			CC_res = np.vstack((CC_res, np.array([np.real(np.fft.ifft(mY))])))

		cou = cou + 1

	return CC_harm, pitch_pyin, CC_res


def CHpS(audio_inp, params, params_ceps, params_R):
	"""
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	# ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']
	w = windows.hann(W)

	# Parameters for the residual cc calculation
	sf = params_R['sf']
	Fr = params_R['Fr']
	Hr = params_R['Hr']
	ceps_coeffs_residual = params_R['ceps_coeffs_residual']

	# F, M, P = HPRM.harmonicModel_pyin(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH=  maxnSines, minSineDur = 0.02, harmDevSlope = 0.1)
	F, M, P = HM.harmonicModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH=  maxnSines,minSineDur = 0.02, minf0 = 140, maxf0 = 1300, f0et = 5, harmDevSlope = 0.01)

	# Cepstral Coefficients Calculation
	CC_harm = np.zeros((F.shape[0], N))
	# CC_res = np.zeros((F.shape[0], Fr))	

	# Extract the pitch from returned F
	pitch_pyin = F[:,0]
	most_likely_pitch = pitch_pyin[pitch_pyin.shape[0]//2]
	pitch_pyin = most_likely_pitch*np.ones(pitch_pyin.shape)

	# Define the new magnitude array based on the modified coefficients sampled from the TAE
	new_M = np.zeros_like(M)
	new_F = np.zeros_like(F)
	new_P = np.zeros_like(P)

	for i in range(pitch_pyin.shape[0]):
		hm_arr = np.arange(maxnSines) + 1
		f0 = pitch_pyin[i]
		new_F[i,:] = f0*hm_arr 

	# Compute the TAE over the Harmonic model

	for i in range(F.shape[0]):
		f0 = pitch_pyin[i]
		if(f0 != 0):
			ceps_coeffs = (int)(1.9*(fs/(2*f0)))
			# ceps_coeffs = N
		else:
			ceps_coeffs = N

		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f = interpolate.interp1d((F[i,:]/fs)*N,M[i,:]/20,kind = 'linear',fill_value = '-6', bounds_error=False)
		# Frequency bins
		# fbins = np.linspace(0,fs/2,N//2)
		fbins = np.arange(N)
		finp = f(fbins)
		zp = np.concatenate((finp[0:N//2],np.array([0]),np.flip(finp[1:N//2])))
		# print(zp.shape)
		specenv,_,_ = fe.calc_true_envelope_spectral(zp,N,thresh,ceps_coeffs,num_iters)
		CC_harm[i,:] = np.real(np.fft.ifft(specenv))
		fp = interpolate.interp1d(np.arange(N//2),specenv[:N//2],kind = 'linear',fill_value = '-6', bounds_error=False)
		new_M[i,:] = 20*fp((new_F[i,:]/params['fs'])*params['N'])


	# Compute the residual by subtracting the above from the original audio
	# Ns = N
	residual = UF.sineSubtraction(audio_inp,N,H,new_F,new_M,P,fs)

	# Compute the framewise TAE for the 'stochastic residual' as an approximation of the residual envelope
	cou = 0
	for frame in FrameGenerator(essentia.array(residual), frameSize = Fr, hopSize = Hr):
		# f0 = pitch_pyin[i]
		# Sub-sample the residual spectrum
		fft_res = np.log10(np.abs(np.fft.fft(frame,Fr)))
		fft_res_ss = resample(fft_res,int(sf*Fr))
		# Resample to original size
		fft_res_ss = resample(fft_res_ss, Fr)
		# ceps_coeffs_residual = ceps_coeffs_residual
		# mY,_,_,_ = fe.calc_true_envelope(frame*windows.hann(Fr),Fr,thresh,ceps_coeffs_residual,num_iters)
		mY,_,_ = fe.calc_true_envelope_spectral(fft_res_ss,Fr,thresh,ceps_coeffs_residual,num_iters)
		# mY = fft_res_ss
		# CC_res[cou,:] = np.real(np.fft.ifft(mY))
		# mY = 20*mY[:Fr//2 + 1]
		if(cou == 0):
			CC_res = np.array([np.real(np.fft.ifft(mY))])
			# spec_res = np.array([fft_res])
			# spec_res_ss = np.array([fft_res_ss])
		else:
			CC_res = np.vstack((CC_res, np.array([np.real(np.fft.ifft(mY))])))
			# spec_res = np.vstack((spec_res,np.array([fft_res])))
			# spec_res_ss = np.vstack((spec_res_ss,np.array([fft_res_ss])))

		cou = cou + 1

	return CC_harm, pitch_pyin, CC_res



######################################################################

# Defining the main Parameters for the modified HpS model
# Run the modified HpR Model and analyze the H,R
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
params_R['sf'] = 0.1

# Specify Path to audio files here
list_audio_files = './'

# Predefine the P-YIN algorithm
# pyin = PitchYinProbabilistic(frameSize = W,hopSize = H,lowRMSThreshold = 0.0001,preciseTime = True)

# Defining the dictionaries to store the pitch values in
# list_notes = ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
# dict_notes = {'L':{k:[] for k in list_notes},'M':{k:[] for k in list_notes},'U':{k:[] for k in list_notes}}

max_ccval_H = (int)((1.9)*(Fs/(2*165)))
max_ccval_R = params_R['ceps_coeffs_residual']

# Defining the dump dictionary
results = {}

print("\nPreprocessing started...\n")
N = len(list_audio_files)
start = time()
p = 0
for it,f in enumerate(list_audio_files):
	progress = (it*100/N)
	if progress > p:
		current = time()
		p+=1
		print('\033[F\033[A\033[K[' + '='*int((progress//5)) + ' '*int((20-(progress//5))) + '] ', int(progress), '%', '\t Time elapsed: ',current-start, ' seconds')


	file_info = f.split('/')[-1].split('_')
	uid = file_info[0].split('-')[0]
	note_played = file_info[0].split('-')[-1]
	octave_played = file_info[1]
	loudness_played = file_info[2]
	style_played = file_info[3][:-4]
	name_af = uid + '_' + octave_played + '_' + note_played + '_' + style_played

	audio_ex = MonoLoader(sampleRate = Fs,filename = f)()

	# Select the relevant portion of audio to compute cc using essentia's silence detection function
	s_3 = StartStopSilence(threshold = -50)
	for frame in FrameGenerator(audio_ex, frameSize = W, hopSize = H):
		sss = s_3(frame)
		start_frame = (int)(sss[0]*H)
		stop_frame = (int)(sss[1]*H)
		ainp = audio_ex[start_frame:stop_frame]

	# pyin_p,vp = pyin(ainp)
	# pyin_p = pyin_p[pyin_p > 0]

	# Extract center +-l frames and average the pitch
	# l = 20
	# centre_frame = len(pyin_p)//2
	# pitch_estimate = np.mean(pyin_p[centre_frame - l:centre_frame + l])

	# if(octave_played == 'M'):
	# 	if(330/2 < pitch_estimate < 320):
	# 		pitch_estimate = 2*pitch_estimate
	# if(octave_played == 'U'):
	# 	if(note_played == 'SaUp'):
	# 		continue
	# 	if(660/2 < pitch_estimate < 640):
	# 		pitch_estimate = 2*pitch_estimate
	# fHz = pitch_estimate

	# Compute the CC's required according to the pitch
	# ccoeff = params['fs']/(2*fHz)
	# params_ceps['ceps_coeffs'] = (int)(1.9*ccoeff)
	# params_ceps['ceps_coeffs_residual'] = (int)(1.9*ccoeff)

	# Compute the cc's
	op_H, fHz, op_R = CHpS(ainp,params,params_ceps,params_R)
	# print(op_H.shape,op_R.shape,fHz.shape)

		# Store cc'c + other relevant parameters in dict
	results[name_af] = {}
	results[name_af]['pitch'] = fHz
	results[name_af]['loudness'] = loudness_played
	results[name_af]['cc'] = op_H[:,:max_ccval_H]
	results[name_af]['res'] = op_R[:,:max_ccval_R]

# Write the inputs to a pickle dump
pickle.dump(results, open('./Carnatic_Processing_Dump_B_CHpS.txt', 'wb'))