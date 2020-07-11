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
from scipy.io.wavfile import write
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import matplotlib.pyplot as pyp
import csv


#################################################################################################################

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
			spec_res = np.array([fft_res])
			spec_res_ss = np.array([fft_res_ss])
		else:
			CC_res = np.vstack((CC_res, np.array([np.real(np.fft.ifft(mY))])))
			spec_res = np.vstack((spec_res,np.array([fft_res])))
			spec_res_ss = np.vstack((spec_res_ss,np.array([fft_res_ss])))

		cou = cou + 1

	return CC_harm, pitch_pyin, CC_res, spec_res, spec_res_ss, residual, F, M, P, new_F, new_M

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

# Pointers to the folder containing the .wav files
main_dir = '/home/krishna/Desktop/IITB/DDP/Datasets/Annotated Data/'
# main_dir = '/home/krishna/Desktop/IITB/DDP/Main/ddp-genmodels/pitch_gen/Carnatic/dataset_analysis/ex1/'
# list_audio_files = glob.glob(main_dir + '*.wav')
instru = 'Behringer'
dir_audio_files = main_dir + instru + '/IndividualNotes/'
list_audio_files = glob.glob(dir_audio_files + '*.wav')

max_ccval_H = (int)((1.9)*(Fs/(2*165)))
max_ccval_R = 50

print("\nProcessing of Carnatic Files started...\n")
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
	name_af = uid + '_' + octave_played + '_' + note_played + '_' + style_played + '_' +loudness_played

	audio_ex = MonoLoader(sampleRate = Fs,filename = f)()

	# Select the relevant portion of audio to compute cc using essentia's silence detection function
	s_3 = StartStopSilence(threshold = -50)
	for frame in FrameGenerator(audio_ex, frameSize = W, hopSize = H):
		sss = s_3(frame)
		start_frame = (int)(sss[0]*H)
		stop_frame = (int)(sss[1]*H)
		ainp = audio_ex[start_frame:stop_frame]

	op_H, fHz, op_R, spec_res, spec_res_ss, og_res, F, M, P, nF, nM = cc_calc_varP(ainp,params,params_ceps,params_R)

	# Reconstruction
	inp_H = op_H[:,:max_ccval_H]
	inp_R = op_R[:,:max_ccval_H]

	new_F = np.zeros((inp_H.shape[0],params['maxnSines']))
	new_M = np.zeros_like(new_F)

	for i in range(fHz.shape[0]):
		hm_arr = np.arange(params['maxnSines']) + 1
		f0 = fHz[i]
		new_F[i,:] = f0*hm_arr 

	for i in range(new_F.shape[0]):

		# Synthesis of Harmonic Component
		fbins = np.linspace(0,params['fs'],params['N'])
		frame = inp_H[i]
		zp = np.pad(frame,[0,params['N'] - len(frame)],mode = 'constant',constant_values=(0, 0))
		zp = np.concatenate((zp[:params['N']//2],np.flip(zp[1:params['N']//2 + 1])))
		specenv = np.real(np.fft.fft(zp))
		fp = interpolate.interp1d(np.arange(params['N']//2),specenv[:params['N']//2],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		new_M[i,:] = 20*fp((new_F[i,:]/params['fs'])*params['N'])

	# print(inp_R.shape[0])
	for i in range(inp_R.shape[0]):
		# Synthesis of Residual Component
		frame = inp_R[i]
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

	hOG = sineModelSynth(F, M, P, W, H, Fs)

	# Harmonic Plots
	l = op_H.shape[0]
	# frame = op[l//2]
	f0 = fHz[l//2]

	inp_slice = ainp[(l//2)*params['H']:(l//2)*params['H'] + params['W']]
	og_spec = np.log10(np.abs(np.fft.fft(inp_slice,params['N'])))
	aud_slice = hOG[(l//2)*params['H']:(l//2)*params['H'] + params['W']]
	R_slice = og_res[(l//2)*params['H']:(l//2)*params['H'] + params['W']]
	fbins = np.arange(params['N'])*(params['fs']/(params['N']))
	f_ind = (int)(0.3*params['N'])
	num_harm = int(fbins[f_ind]/f0)

	# OG_Spec
	pyp.figure()
	pyp.plot(fbins[:f_ind + 1],20*og_spec[:f_ind + 1],color = 'gray',label = 'Residual Magnitude Spectrum')
	pyp.ylim([np.min(20*og_spec[:f_ind + 1]) - 5,np.max(20*og_spec[:f_ind + 1]) + 5])
	pyp.axis('off')
	pyp.savefig('./magplots/' + name_af + '_og.pdf',bbox_inches='tight',pad_inches = 0)
	pyp.close()

	# OG Spec + Harmonic + TAE
	og_spec_H = 20*np.log10(np.abs(np.fft.fft(aud_slice,params['N'])))
	og_spec_H = og_spec_H - (np.max((og_spec_H)) - np.max((M[l//2])))
	pyp.figure()
	# pyp.title(name_af)
	pyp.plot(fbins[:f_ind + 1],og_spec_H[:f_ind + 1],color = 'red',label = 'Harmonic Magnitude Spectrum')
	# pyp.vlines(F[l//2][:num_harm - 1],-100,M[l//2][:num_harm - 1],color = 'red')
	pyp.plot(F[l//2][:num_harm - 1],M[l//2][:num_harm - 1],'k-', label = 'Harmonic Subsampled Spectrum')
	frame = inp_H[l//2]
	zp = np.pad(frame,[0,params['N'] - len(frame)],mode = 'constant',constant_values=(0, 0))
	zp = np.concatenate((zp[:params['N']//2],np.flip(zp[1:params['N']//2 + 1])))
	specenv_H = np.real(np.fft.fft(zp))
	# pyp.plot(fbins[:f_ind + 1],20*specenv_H[:f_ind + 1],'g-',label = 'Harmonic TAE')
	# pyp.xlabel('Frequency(Hz)')
	# pyp.ylabel('Magnitude(dB)')
	# pyp.legend(loc = 'best')
	pyp.ylim([np.min(og_spec_H[:f_ind + 1]) - 5,np.max(og_spec_H[:f_ind + 1]) + 5])
	pyp.axis('off')
    # pyp.savefig('./icassp/' + name + '_inp_tae.pdf',bbox_inches='tight',pad_inches = 0)
	pyp.savefig('./magplots/' + name_af + '_H1.pdf',bbox_inches='tight',pad_inches = 0)
	pyp.close()

	pyp.figure()
	pyp.vlines(F[l//2][:num_harm - 1],np.min(og_spec_H[:f_ind + 1]) - 5,M[l//2][:num_harm - 1],color = 'red')
	pyp.plot(fbins[:f_ind + 1],20*specenv_H[:f_ind + 1],'k-',label = 'Harmonic TAE')
	pyp.ylim([np.min(og_spec_H[:f_ind + 1]) - 5,np.max(og_spec_H[:f_ind + 1]) + 5])
	pyp.axis('off')
	pyp.savefig('./magplots/' + name_af + '_H2.pdf',bbox_inches='tight',pad_inches = 0)
	pyp.close()

	pyp.figure()
	# pyp.vlines(F[l//2][:num_harm - 1],np.min(og_spec_H[:f_ind + 1]) - 5,M[l//2][:num_harm - 1],color = 'red')
	pyp.plot(fbins[:f_ind + 1],20*specenv_H[:f_ind + 1],'k-',label = 'Harmonic TAE')
	pyp.plot(nF[l//2][:num_harm - 1],nM[l//2][:num_harm - 1],'x',color = 'red')
	# pyp.ylim([np.min(og_spec_H[:f_ind + 1]) - 5,np.max(og_spec_H[:f_ind + 1]) + 5])
	pyp.ylim([-100,20])
	# pyp.axis('off')
	pyp.savefig('./resplots/' + name_af + '_H2.png',bbox_inches='tight',pad_inches = 0)
	pyp.close()

	with open('./resplots/' + name_af + '_H.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(['f','ogH','taeH'])
		for kp in range(fbins.shape[0]):
			csvwriter.writerow([fbins[kp],og_spec_H[kp],20*specenv_H[kp]])

	with open('./resplots/' + name_af + '_H_harmlocs.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(['nF','nM'])
		for kp in range(nF[l//2].shape[0]):
			csvwriter.writerow([nF[l//2][kp],nM[l//2][kp]])

	# OG Res + SS + TAE
	fbins = np.arange(params_R['Fr'])*(params['fs']/(params_R['Fr']))
	f_ind = (int)(0.3*params_R['Fr'])
	og_spec_R = 20*spec_res[l//2]
	og_spec_R_ss = 20*spec_res_ss[l//2]
	pyp.figure()
	# pyp.title(name_af)
	pyp.plot(fbins[:f_ind + 1],og_spec_R[:f_ind + 1],color = 'red',label = 'Residual Magnitude Spectrum')
	pyp.plot(fbins[:f_ind + 1],og_spec_R_ss[:f_ind + 1],color = 'black',label = 'Residual Subsampled Spectrum')
	# pyp.vlines(F[l//2][:num_harm - 1],-100,M[l//2][:num_harm - 1],color = 'red')
	# pyp.plot(F[l//2][:num_harm - 1],M[l//2][:num_harm - 1],'x',color = 'blue')
	frame = inp_R[l//2]
	zp = np.pad(frame,[0,params_R['Fr'] - len(frame)],mode = 'constant',constant_values=(0, 0))
	zp = np.concatenate((zp[:params_R['Fr']//2],np.flip(zp[1:params_R['Fr']//2 + 10])))
	specenv_R = np.real(np.fft.fft(zp))
	# pyp.plot(fbins[:f_ind + 1],20*specenv_R[:f_ind + 1],'g-',label = 'Residual TAE')
	# pyp.xlabel('Frequency(Hz)')
	# pyp.ylabel('Magnitude(dB)')
	# pyp.legend(loc = 'best')
	pyp.ylim([np.min(og_spec_R[:f_ind + 1]) - 5,np.max(og_spec_R[:f_ind + 1]) + 5])
	pyp.axis('off')
	pyp.savefig('./magplots/' + name_af + '_R1.pdf',bbox_inches='tight',pad_inches = 0)
	pyp.close()

	pyp.figure()
	pyp.plot(fbins[:f_ind + 1],og_spec_R[:f_ind + 1],color = 'red',label = 'Residual Magnitude Spectrum')
	pyp.plot(fbins[:f_ind + 1],20*specenv_R[:f_ind + 1],'k-',label = 'Residual TAE')
	pyp.ylim([np.min(og_spec_R[:f_ind + 1]) - 5,np.max(og_spec_R[:f_ind + 1]) + 5])
	pyp.axis('off')
	pyp.savefig('./magplots/' + name_af + '_R2.pdf',bbox_inches='tight',pad_inches = 0)
	pyp.close()

	pyp.figure()
	# pyp.plot(fbins[:f_ind + 1],og_spec_R[:f_ind + 1],color = 'red',label = 'Residual Magnitude Spectrum')
	pyp.plot(fbins[:f_ind + 1],20*specenv_R[:f_ind + 1],'k-',label = 'Residual TAE')
	# pyp.ylim([np.min(og_spec_R[:f_ind + 1]) - 5,np.max(og_spec_R[:f_ind + 1]) + 5])
	pyp.ylim([-100,20])
	# pyp.axis('off')
	pyp.savefig('./resplots/' + name_af + '_R2.png',bbox_inches='tight',pad_inches = 0)
	pyp.close()

	with open('./resplots/' + name_af + '_R.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(['f','ogR','taeR'])
		for kp in range(fbins.shape[0]):
			csvwriter.writerow([fbins[kp],og_spec_R[kp],20*specenv_R[kp]])

	
	# Add the residual to the harmonic to obtain the audio
	ltw = min(len(harmonic),len(stochastic_residual))

	hps_recon_audio = harmonic[:ltw] + stochastic_residual[:ltw]

	# Save the reconstructed audio
	# write(filename = './CHpS/' + name_af + '_reconH.wav', rate = params['fs'], data = harmonic.astype('float32'))
	# write(filename = './CHpS/' + name_af + '_reconR.wav', rate = params['fs'], data = stochastic_residual.astype('float32'))
	# write(filename = './CHpS/' + name_af + '_recon.wav', rate = params['fs'], data = hps_recon_audio.astype('float32'))
	# write(filename = './CHpS/' + name_af + '_originalR.wav', rate = params['fs'], data = og_res.astype('float32'))
	# write(filename = './CHpS/' + name_af + '_original.wav', rate = params['fs'], data = ainp.astype('float32'))