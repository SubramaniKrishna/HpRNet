# Dependencies
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as pyp
import numpy as np
from datetime import datetime
from time import time
import glob
import ast
import librosa
import os

# Importing our model
import sys
sys.path.append('../Dependencies/')
sys.path.append('../Dependencies/models/')
from vae_krishna import cVAE_synth
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
from essentia.standard import MonoLoader
from scipy.io.wavfile import write
import sampling_synth as ss
from scipy.signal import windows
from scipy import interpolate
import pickle

# Import custom Carnatic Dataloader
from carnatic_DL import *
from HpRNet import HPRNet


# Load the data
datafile = '../dataset_analysis/Carnatic_Processing_Dump_B_CHpS.txt'
data = pickle.load(open(datafile, 'rb'))
# Dataset conditions when testing
# ['L','M','U']
list_octave = ['U']
# ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
list_notes = ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2']
# ['So','Lo']
list_loudness = ['So', 'Lo']
# ['Sm','At']
list_style = ['Sm']



# Load the parameters from the strig of the .pth file
dir_files = './paramshprnet/'
list_pth_files = glob.glob(dir_files + '*.pth')

# Display the available parameter files
print('Available state files to load are(for HpRNet) : ')
for it,i in enumerate(list_pth_files):
	print(it,i.split('/')[-1][:-4])

# Choose the .pth file
idx = (int)(input('Choose the input index'))
print(list_pth_files[idx].split('_')[1])
# Load the parameters into a list
list_params = ast.literal_eval(list_pth_files[idx].split('_')[1])
# print(list_params)
file_load_HpRNet = list_pth_files[idx]


# list params, make the model and load the wights from the .pth file
# Fix device here(currently cpu)
device = 'cpu'
# device = 'cuda'
# Defining the model architecture
dim_cc = list_params[0]
flag_cond = list_params[1]
layer_dims_enc = list_params[2]
latent_dims = list_params[3]
layer_dims_dec = list_params[4]
num_cond = list_params[5]

# print(layer_dims_encH,latent_dimsH,layer_dims_decH)

HpRNet = HPRNet(flag_cond = flag_cond, layer_dims_enc = layer_dims_enc, layer_dims_dec = layer_dims_dec, latent_dims = latent_dims, num_cond = num_cond, device = device)
HpRNet.load_state_dict(torch.load(file_load_HpRNet,map_location = 'cpu'))


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
params['nH'] = params['maxnSines']
params_ceps = {}
params_ceps['thresh'] = 0.1
params_ceps['num_iters'] = 1000
params_R = {}
params_R['Fr'] = 512
params_R['Hr'] = 256
params_R['ceps_coeffs_residual'] = 512

# Pitch normalizing factor
pnf = 660

# Dimensionality of cepstral input to network (defined by lowest frequency, update accordingly!)
cc_keep_H = (int)(1.9*(44100/(2*660)))
cc_keep_R = 50

dir_save = './net_recon/HpRNet_' + list_pth_files[idx].split('_')[0][-7:-1] + '/'
try: 
	os.mkdir(dir_save) 
except OSError as error: 
    print('Dir Exists')

t = 0
for k in data.keys():
	print (t+1,'/',len(data.keys()))
	t = t + 1
	
	temp_file = data[k]
	p = temp_file['pitch']
	cc_H = temp_file['cc']
	cc_R = temp_file['res']

	uid = k.split('_')[0]
	octave = k.split('_')[1]
	note = k.split('_')[2]
	style = k.split('_')[3]
	loudness = temp_file['loudness']

	if(octave in list_octave and note in list_notes and loudness in list_loudness and style in list_style):
		p = torch.FloatTensor(p)
		lenP = p.shape[0]
		inpH = torch.FloatTensor(cc_H[:lenP,:cc_keep_H])
		ccR = torch.FloatTensor(cc_R)
		Z = torch.zeros((inpH.shape[0],cc_keep_H - cc_keep_R))
		inpR = torch.cat((ccR.float()[:lenP,:cc_keep_R],Z.float()),1)

		if(flag_cond == True):
			ccH_recon_cVAE,ccR_recon_cVAE = HpRNet.return_HR(inpH,inpR,(p.float()/pnf).view(-1,1))
		else:
			ccH_recon_cVAE,ccR_recon_cVAE = HpRNet.return_HR(inpH,inpR)

		a_og = ss.recon_samples_ls(matrix_ceps_coeffs = cc_H.T, midi_pitch = p, params = params, choice_f = 1)
		sr_og = ss.recon_residual_ls(matrix_ceps_coeffs = cc_R.T, params = params_R)
		a_recon_cVAE = ss.recon_samples_ls(matrix_ceps_coeffs = ccH_recon_cVAE.detach().numpy().T, midi_pitch = p, params = params, choice_f = 1)
		sr_recon_cVAE = ss.recon_residual_ls(matrix_ceps_coeffs = ccR_recon_cVAE.detach().numpy().T,params = params_R)

		# Add the residual to the harmonic to obtain the audio
		ltw = min(len(a_og),len(sr_og))
		hps_og = a_og[:ltw] + sr_og[:ltw]
		ltw = min(len(a_recon_cVAE),len(sr_recon_cVAE))
		hps_recon_cVAE = a_recon_cVAE[:ltw] + sr_recon_cVAE[:ltw]

		# Save the files as .wav files
		write(filename = dir_save + '_' + str(k) + '_' + loudness + '_og_H.wav', rate = params['fs'], data = a_og.astype('float32'))
		write(filename = dir_save + '_' + str(k) + '_' + loudness + '_og_R.wav', rate = params['fs'], data = sr_og.astype('float32'))
		write(filename = dir_save + '_' + str(k) + '_' + loudness + '_og_HpR.wav', rate = params['fs'], data = hps_og.astype('float32'))
		write(filename = dir_save + '_' + str(k) + '_' + loudness + '_recon_H.wav', rate = params['fs'], data = a_recon_cVAE.astype('float32'))
		write(filename = dir_save + '_' + str(k) + '_' + loudness + '_recon_R.wav', rate = params['fs'], data = sr_recon_cVAE.astype('float32'))
		write(filename = dir_save + '_' + str(k) + '_' + loudness + '_recon_HpR.wav', rate = params['fs'], data = hps_recon_cVAE.astype('float32'))