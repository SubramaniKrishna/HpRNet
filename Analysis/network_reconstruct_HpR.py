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


# Load the data
datafile = '../dataset_analysis/Carnatic_Processing_Dump_B_CHpS.txt'
data = pickle.load(open(datafile, 'rb'))
# Dataset conditions when testing
# ['L','M','U']
list_octave = ['U']
# ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
list_notes = ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2']
# ['So','Lo']
list_loudness = ['So','Lo']
# ['Sm','At']
list_style = ['Sm']



# Load the parameters from the strig of the .pth file
dir_files = './paramshpr/'
list_pth_files = glob.glob(dir_files + '*.pth')

# Display the available parameter files
print('Available state files to load are(for Harmonic cVAE) : ')
for it,i in enumerate(list_pth_files):
	print(it,i.split('/')[-1][:-4])

# Choose the .pth file
idx = (int)(input('Choose the input index'))
print(list_pth_files[idx].split('_')[1])
# Load the parameters into a list
list_params = ast.literal_eval(list_pth_files[idx].split('_')[1])
hid = list_pth_files[idx].split('_')[0][-7:-1]
# print(list_params)
file_load_HCVAE = list_pth_files[idx]


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

# print(layer_dims_encH,latent_dimsH,layer_dims_decH)

cVAE_H = cVAE_synth(flag_cond = flag_cond_H, layer_dims_enc = layer_dims_encH, layer_dims_dec = layer_dims_decH, latent_dims = latent_dimsH, num_cond = num_cond, device = device)
cVAE_H.load_state_dict(torch.load(file_load_HCVAE,map_location = 'cpu'))

# Display the available parameter files
print('Available state files to load are(for Residual cVAE) : ')
for it,i in enumerate(list_pth_files):
	print(it,i.split('/')[-1][:-4])

# Choose the .pth file
idx = (int)(input('Choose the input index'))
# print(list_pth_files[idx])
# Load the parameters into a list
list_params = ast.literal_eval(list_pth_files[idx].split('_')[1])
rid = list_pth_files[idx].split('_')[0][-7:-1]
file_load_RCVAE = list_pth_files[idx]


# list params, make the model and load the wights from the .pth file
# Fix device here(currently cpu)
device = 'cpu'
# device = 'cuda'
# Defining the model architecture
dim_cc = list_params[0]
flag_cond_R = list_params[1]
layer_dims_encR = list_params[2]
latent_dimsR = list_params[3]
layer_dims_decR = list_params[4]
num_cond = list_params[5]


# print(layer_dims_encR,latent_dimsR,layer_dims_decR)

cVAE_R = cVAE_synth(flag_cond = flag_cond_R, layer_dims_enc = layer_dims_encR, layer_dims_dec = layer_dims_decR, latent_dims = latent_dimsR, num_cond = num_cond, device = device)
cVAE_R.load_state_dict(torch.load(file_load_RCVAE,map_location = 'cpu'))




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
params_R['ceps_coeffs_residual'] = 50

# Pitch normalizing factor
pnf = 660

# Dimensionality of cepstral input to network (defined by lowest frequency, update accordingly!)
cc_keep_H = (int)(1.9*(44100/(2*660)))
cc_keep_R = 50

dir_save = './net_recon/INet_' + hid + '_' + rid + '/'
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
		inpR = torch.FloatTensor(cc_R[:lenP,:cc_keep_R])
		# print(inpH.shape,inpR.shape,p.shape)
		if(flag_cond_H == True and flag_cond_R == True):
			ccH_recon_cVAE,mu,sig,ztot = cVAE_H.forward(inpH,(p.float()/pnf).view(-1,1))
			ccR_recon_cVAE,mu,sig,ztot = cVAE_R.forward(inpR,(p.float()/pnf).view(-1,1))
		elif(flag_cond_H == False and flag_cond_R == True):
			ccH_recon_cVAE,mu,sig,ztot = cVAE_H.forward(inpH)
			ccR_recon_cVAE,mu,sig,ztot = cVAE_R.forward(inpR,(p.float()/pnf).view(-1,1))
		elif(flag_cond_H == True and flag_cond_R == False):
			ccH_recon_cVAE,mu,sig,ztot = cVAE_H.forward(inpH,(p.float()/pnf).view(-1,1))
			ccR_recon_cVAE,mu,sig,ztot = cVAE_R.forward(inpR)
		else:
			ccH_recon_cVAE,mu,sig,ztot = cVAE_H.forward(inpH)
			ccR_recon_cVAE,mu,sig,ztot = cVAE_R.forward(inpR)
			
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