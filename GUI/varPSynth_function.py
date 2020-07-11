# Dependencies
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from time import time
import glob
import ast
import librosa
import os
from essentia.standard import MonoLoader, StartStopSilence, FrameGenerator, PitchYinProbabilistic, PitchYin
import essentia


# Importing our model
import sys
sys.path.append('../Dependencies/')
sys.path.append('../Dependencies/models/')
from vae_krishna import cVAE_synth
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
import stft as STFT
from essentia.standard import MonoLoader
from scipy.io.wavfile import write
import sampling_synth as ss
from scipy.signal import windows
from scipy import interpolate
import pickle

# Import custom Carnatic Dataloader
from carnatic_DL import *
lp_x = []
lp_y = []
class LineBuilder:
	def __init__(self, line):
		self.line = line
		self.xs = list(line.get_xdata())
		self.ys = list(line.get_ydata())
		self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

	def __call__(self, event):
		print('click', event)
		if event.inaxes!=self.line.axes: return
		lp_x.append(event.xdata)
		lp_y.append(event.ydata)
		# print(list_points_clicked)
		self.xs.append(event.xdata)
		self.ys.append(event.ydata)
		self.line.set_data(self.xs, self.ys)
		self.line.figure.canvas.draw()

def main(pH,choice,dur,Vcf,Av,Vf,pitchF):
	# Harmonic .pth file
	list_params = ast.literal_eval(pH.split('/')[-1].split('_')[1])
	hid = pH.split('/')[-1].split('_')[0][-7:-1]
	# print(list_params)
	file_load_HCVAE = pH


	# list params, make the model and load the wights from the .pth file
	# Fix device here(currently cpu)
	device = 'cpu'
	# device = 'cuda'
	# Defining the model architecture
	dim_cc = list_params[0]
	flag_cond_H = list_params[1]
	layer_dims_encH = list_params[2]
	latent_dimsH = list_params[3]
	layer_dims_decH = list_params[4]
	num_cond = list_params[5]

	cVAE_H = cVAE_synth(flag_cond = flag_cond_H, layer_dims_enc = layer_dims_encH, layer_dims_dec = layer_dims_decH, latent_dims = latent_dimsH, num_cond = num_cond, device = device)
	cVAE_H.load_state_dict(torch.load(file_load_HCVAE,map_location = 'cpu'))

	# Defining the main Parameters for the modified HpS model
	# Run the modified HpR Model and analyze the H,R
	params = {}
	params['fs'] = 44100
	Fs = params['fs']
	params['W'] = 1024
	W = params['W']
	w = windows.hann(W)
	params['N'] = 2048
	N = params['N']
	params['H'] = 256
	H = params['H']
	params['t'] = -120
	params['maxnSines'] = 150
	params['nH'] = params['maxnSines']
	params_ceps = {}
	params_ceps['thresh'] = 0.1
	params_ceps['num_iters'] = 1000
	params_R = {}
	params_R['Fr'] = 512
	params_R['Hr'] = 256
	params_R['ceps_coeffs_residual'] = 512

	# Pitch normalizing factor
	pnf = (int)(1.9*(44100/(2*dim_cc)))

	# Dimensionality of cepstral input to network (defined by lowest frequency, update accordingly!)
	cc_keep = (int)(1.9*(44100/(2*pnf)))

	# Dimensionality of latent space
	ld = latent_dimsH
	# Specify the duration of the note in seconds
	dur_n = dur
	nf = (int)(params['fs']/(params['H'])*dur_n)
	"""
	You can take the pitch inputs in two ways (setting the choice variable to '0' or '1')
	choice = 0:
		Here, you can manually input the pitch contour. Just specify the start and end frequencies. A matplotlib plot will pop, asking you to click at points.
		Each point you click is a (pitch,time) pair, and the more points you add, the finer the sampling. The pitch contour will be formed by interpolating appropriately
		Once you have specified the contour, close the matplotlib popup window.
	choice = 1:
		Here, a single note with vibrato is generated. You can specify the vibrato parameters as needed.
	choice = 2:
		Custom pitch track, will be extracted from a .wav file using pyin: the duration will be decided by the length of the pitch track + hop size
	An audio file will be saved in the specified directory
	"""

	# if(choice == "DrawP"):
	# 	# ______________________________________________________________________________________________________________________________________
	# 	# Choice = 0;
	# 	# Obtaining the Pitch Contour by drawing on matplotlib
	# 	# Obtaining the Contour by passing (pitch,time) coordinates and linearly interpolating the frequencies in between

	# 	# Starting Frequency (Specify)
	# 	f_start = startF
	# 	# Ending frequency
	# 	f_end = endF


	# 	fig = plt.figure()
	# 	ax = fig.add_subplot(111)
	# 	ax.set_title('Click to select the pitch points (they will be linearly interpolated')
	# 	plt.ylim(f_start,f_end)
	# 	plt.xlim(0,dur_n)
	# 	line, = ax.plot([0], [f_start])  # empty line
	# 	linebuilder = LineBuilder(line)
	# 	plt.show()
	# 	plt.close()


	# 	# Specify array containing the time instants and pitches
	# 	# The pitch contour will be formed by linearly interpolating
	# 	# array_time_instants = np.array([0.5,1.1,2.3,2.5,2.8])
	# 	# array_frequencies = np.array([260,290,250,350,400])
	# 	array_time_instants = np.array(lp_x)
	# 	array_frequencies = np.array(lp_y)
	# 	num_points = array_frequencies.shape[0] 

	# 	# Append the start and end frequencies to the main frequency array. Do same with time(start -> 0 and stop-> duration specified)
	# 	array_frequencies = np.insert(array_frequencies,[0,num_points],[f_start,f_end])
	# 	array_time_instants = np.insert(array_time_instants,[0,num_points],[0,dur_n])
	# 	# print(array_frequencies)
	# 	# print(array_time_instants)
	# 	#Assuming that spacing between all frequencies is uniform (i.e. more the frequencies specified, more dense the sampling)
	# 	# nbf = (int)(nf/num_points)
	# 	fcontour_Hz = np.zeros(nf)

	# 	for i in range(0,len(array_frequencies) - 1):
	# 		s = array_time_instants[i]
	# 		e = array_time_instants[i+1]
	# 		# print(s,e)
	# 		s = (int)((s/dur_n)*nf)
	# 		e = (int)((e/dur_n)*nf)
	# 		nbf = (e - s)
	# 		# print(s,e)
	# 		fr = np.linspace(array_frequencies[i],array_frequencies[i+1],nbf)
	# 		fcontour_Hz[s:e] = fr
	# # print(fcontour_Hz)

	if(choice == 'VibratoP'):
		# ____________________________________________________________________________________________________________________________________
		# Choice = 1;
		# Generating a note with Vibrato (Frequency Modulation)
		# Vibrato pitch contour in Hz
		# Center Frequency in MIDI
		# p = 69
		# Obtain f_c by converting the pitch from MIDI to Hz
		f_Hz = Vcf

		# Vibrato depth(1-2% of f_c)
		Av = (Av/100.0)*f_Hz

		# Vibrato frequency(generally 5-10 Hz)
		fV_act = Vf
		# Sub/sampling the frequency according to the Hop Size
		f_v = 2*np.pi*((fV_act*params['H'])/(params['fs']))

		# Forming the contour
		# The note will begin with a sustain pitch, and then transition into a vibrato
		# Specify the fraction of time the note will remain in sustain
		frac_sus = 0.25
		fcontour_Hz = np.concatenate((f_Hz*np.ones((int)(nf*frac_sus) + 1),f_Hz + Av*np.sin(np.arange((int)((1-frac_sus)*nf))*f_v),(f_Hz*np.ones((int)(nf*frac_sus) + 1))))

	else:
		# Predefine the P-YIN algorithm
		pyin = PitchYinProbabilistic(frameSize = W,hopSize = H,lowRMSThreshold = 0.0001,preciseTime = True)
		audio_ex = MonoLoader(sampleRate = Fs,filename = pitchF)()
		pyin_p,vp = pyin(audio_ex)
		fcontour_Hz = pyin_p

	# Filter out pitch contour from the <0 frequencies
	fcontour_Hz[fcontour_Hz < 0] = 0

	# Obtain a trajectory in the latent space using a random walk
	nf = fcontour_Hz.shape[0]
	z_ss = 0.0001*ss.rand_walk(np.zeros(ld), 0.001, nf)
	z_ss1 = torch.FloatTensor(z_ss.T)
	cond_inp = torch.FloatTensor(fcontour_Hz)
	cond_inp = cond_inp.float()/pnf
	# print(z_ss1.shape,cond_inp.shape)
	# Sample from the CVAE latent space
	s_z_X = cVAE_H.sample_latent_space(z_ss1,cond_inp.view(-1,1))
	cc_network = s_z_X.data.numpy().squeeze()

	dir_gen_audio = './audioout/'
	a_gen_cVAE = ss.recon_samples_ls(matrix_ceps_coeffs = cc_network.T, midi_pitch = fcontour_Hz, params = params,choice_f = 1)
	mX_gen, pX_gen = STFT.stftAnal(a_gen_cVAE, w, N, H)
	# if(choice == 'DrawP'):
		# write(filename = dir_gen_audio + pitchF + '_genH.wav', rate = params['fs'], data = a_gen_cVAE.astype('float32'))
	# else:
	write(filename = dir_gen_audio + pitchF.split('/')[-1][:-4] + '_genH.wav', rate = params['fs'], data = a_gen_cVAE.astype('float32'))
	# print('Plot Specgram')
	# Plot the Spectrogram of the Generated Audio
	# Plotting the spectrograms
	plt.figure('Spectrogram',figsize=(12, 9))
	# frequency range to plot
	maxplotfreq = 10000.0

	# plot original magnitude spectrogram
	plt.title('Generated Audio Magnitude Spectrogram')
	numFrames = int(mX_gen[:,0].size)
	frmTime = H*np.arange(numFrames)/float(Fs)
	binFreq = Fs*np.arange(N*maxplotfreq/Fs)/N
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX_gen[:,:int(N*maxplotfreq/Fs+1)]))
	plt.plot(frmTime,fcontour_Hz,'r')
	plt.xlabel('time (s)')
	plt.ylabel('frequency (Hz)')
	plt.autoscale(tight=True)
	plt.colorbar()
	plt.tight_layout()
	plt.ion()
	plt.show()

if __name__ == "__main__":
	main()
