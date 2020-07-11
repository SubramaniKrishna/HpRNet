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
import copy
import json

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

# Loading the Networks
from HpRNet import HPRNet


# Load the data
datafile = '../dataset_analysis/Carnatic_Processing_Dump_B_CHpS.txt'
data = pickle.load(open(datafile, 'rb'))


# Remember to set the correct conditions during evaluation (no data from the train should be included)
# Dataset conditions when testing
# ['L','M','U']
list_octave = ['U']
# ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
list_notes = ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
# ['So','Lo']
list_loudness = ['So','Lo']
# ['Sm','At']
list_style = ['Sm']

# List of Algos to evaluate
list_algos = ['INet','HpRNet','ConcatNet']

# Store corresponding directories of .pth files in a list
dir_INet = './paramshpr/'
dir_HpRNet = './paramshprnet/'
dir_ConcatNet = './paramsconcat/'
dir_algos = [glob.glob(dir_INet + '*.pth'),glob.glob(dir_HpRNet + '*.pth'),glob.glob(dir_ConcatNet + '*.pth')]

# Defining the directory dict to store the locations of the .pth files
dd = {k:dir_algos[it] for it,k in enumerate(list_algos)}
# Defining the error dict
ed = {k:{} for k in list_algos}
# Latent space dict
ld = {k:{} for k in list_algos}

# The below can either be global or local. If local, insert the appropriate values before processing the evaluation in the for loop below
pnf = 660
cc_keep = (int)(1.9*(44100/(2*660)))
# Commmon params
device = 'cpu'

dim_cc_H = (int)(1.9*(44100/(2*660)))
dim_cc_R = 50

for k in list_algos:
	# print("Algorithm Currently Being Processed : ",k)
	print('New Algo : ', k)
	for f in dd[k]:
		if(k == 'INet' and f[-5] == 'R'):
			continue
		print('New .pth file')
		# print(f)
		uid = f.split('/')[-1].split('_')[0].split('(')[-1][:-1]
		ed[k][uid] = {}
		ld[k][uid] = {}

		for l in list_notes:
			print(k,uid,l)
			num_frames = 50000
			ds = CarnaticDL(filename = datafile, num_frames = num_frames, list_octave = list_octave, list_notes = [l], list_loudness = list_loudness, list_style = list_style)
			p_arr = []
			cc_H = []
			cc_R = []
			for dp in ds:
				p_arr.append(dp[1].item())
				cc_H.append(dp[3].data.numpy())
				cc_R.append(dp[4].data.numpy())
			p_arr = np.array(p_arr)
			cc_H = np.array(cc_H)
			cc_R = np.array(cc_R)
			temp_H = torch.tensor(cc_H[:,:dim_cc_H])
			inp_cc_H = temp_H.contiguous().to(device, non_blocking=True)
			temp_R = torch.tensor(cc_R[:,:dim_cc_R])
			inp_cc_R = temp_R.contiguous().to(device, non_blocking=True)
			pitch = torch.tensor(p_arr)
			c = (pitch.float()/pnf).to(device, non_blocking=True)
			f0 = c.view(-1,1)

			# Extract network parameters from file name
			list_params = ast.literal_eval(f.split('_')[1])
			# Check algorithm, perform loading according to that
			if(k == 'INet'):
				dim_cc = dim_cc_H
				flag_cond = list_params[1]
				latent_dimsH = list_params[3]
				latent_dimsR = list_params[3]
				layer_dims_encH = copy.deepcopy(list_params[2])
				layer_dims_encR = copy.deepcopy([dim_cc_R,50])
				layer_dims_decH = copy.deepcopy(list_params[4])
				layer_dims_decR = copy.deepcopy([latent_dimsR,50,dim_cc_R])
				num_cond = list_params[5]

				dir_networkH = './paramshpr/' + f.split('/')[-1].split('_')[0] + '_' + str([dim_cc_H, flag_cond, layer_dims_encH, latent_dimsH, layer_dims_decH, num_cond]) + f[-39:-5] + 'H.pth'
				dir_networkR = './paramshpr/' + f.split('/')[-1].split('_')[0] + '_' + str([dim_cc_R, False, layer_dims_encR, latent_dimsR, layer_dims_decR, 0]) + f[-39:-5] + 'R.pth'

				# print(dir_networkH)
				# print(dir_networkR)

				cVAE_H = cVAE_synth(flag_cond = flag_cond, layer_dims_enc = layer_dims_encH, layer_dims_dec = layer_dims_decH, latent_dims = latent_dimsH, num_cond = num_cond, device = device)
				cVAE_H.load_state_dict(torch.load(dir_networkH,map_location = 'cpu'))

				cVAE_R = cVAE_synth(flag_cond = False, layer_dims_enc = layer_dims_encR, layer_dims_dec = layer_dims_decR, latent_dims = latent_dimsR, num_cond = 0, device = device)
				cVAE_R.load_state_dict(torch.load(dir_networkR,map_location = 'cpu'))

				if(flag_cond == True):
					ccH_recon_cVAE,mu,sig,zH = cVAE_H(inp_cc_H.float(),f0.float())
					ccR_recon_cVAE,mu,sig,zR = cVAE_R(inp_cc_R.float(),f0.float())
				else:
					ccH_recon_cVAE,mu,sig,zH = cVAE_H(inp_cc_H.float())
					ccR_recon_cVAE,mu,sig,zR = cVAE_R(inp_cc_R.float())

			elif(k == 'HpRNet'):
				dim_cc = list_params[0]
				flag_cond = list_params[1]
				layer_dims_enc = list_params[2]
				latent_dims = list_params[3]
				layer_dims_dec = list_params[4]
				num_cond = list_params[5]

				HpRNet = HPRNet(flag_cond = flag_cond, layer_dims_enc = layer_dims_enc, layer_dims_dec = layer_dims_dec, latent_dims = latent_dims, num_cond = num_cond, device = device)
				HpRNet.load_state_dict(torch.load(f,map_location = 'cpu'))

				Z = torch.zeros((temp_H.shape[0],dim_cc_H - dim_cc_R))
				inp_cc_R = torch.cat((inp_cc_R.float()[:,:dim_cc_R],Z.float()),1)

				if(flag_cond == True):
					ccH_recon_cVAE,ccR_recon_cVAE = HpRNet.return_HR(inp_cc_H.float(),inp_cc_R.float(),f0.float())
					xs_recon, mu_s, sigma_covar_s, zs, xd_recon, mu_d, sigma_covar_d, zd = HpRNet(inp_cc_H.float(),inp_cc_R.float(),f0.float())
				else:
					ccH_recon_cVAE,ccR_recon_cVAE = HpRNet.return_HR(inp_cc_H.float(),inp_cc_R.float())
					xs_recon, mu_s, sigma_covar_s, zs, xd_recon, mu_d, sigma_covar_d, zd = HpRNet(inp_cc_H.float(),inp_cc_R.float())

				zH = 0.5*(zs + zd)
				zR = 0.5*(zs - zd)
			else:
				dim_cc = list_params[0]
				flag_cond = list_params[1]
				layer_dims_enc = list_params[2]
				latent_dims = list_params[3]
				layer_dims_dec = list_params[4]
				num_cond = list_params[5]

				cVAE_C = cVAE_synth(flag_cond = flag_cond, layer_dims_enc = layer_dims_enc, layer_dims_dec = layer_dims_dec, latent_dims = latent_dims, num_cond = num_cond, device = device)
				cVAE_C.load_state_dict(torch.load(f,map_location = 'cpu'))

				inpNet = torch.cat((inp_cc_H,inp_cc_R), dim = 1)

				if(flag_cond == True):
					xC_recon, mu, sigma_covar, z = cVAE_C.forward(inpNet.float(),f0.float())
				else:
					xC_recon, mu, sigma_covar, z = cVAE_C.forward(inpNet.float())
				zH = z[:,:latent_dims//2]
				zR = z[:,latent_dims//2:latent_dims]
				# print(zH.shape,zR.shape,z.shape)
				ccH_recon_cVAE = xC_recon[:,:cc_keep]
				ccR_recon_cVAE = xC_recon[:,cc_keep:2*cc_keep]

			# Calculating the error and storing it in the appropriate dictionary entry
			e_H = torch.nn.functional.mse_loss(inp_cc_H.float(), ccH_recon_cVAE.float(), reduction='sum').item()
			# Normalize Error by dividing by total number of data points
			# print(inp_cc_H.shape)
			e_H = e_H/inp_cc_H.shape[0]

			e_R = torch.nn.functional.mse_loss(inp_cc_R.float(), ccR_recon_cVAE.float(), reduction='sum').item()
			# Normalize Error by dividing by total number of data points
			# print(inp_cc_R.shape)
			e_R = e_R/inp_cc_R.shape[0]

			ed[k][uid][l] = [e_H,e_R]
			# print(zH.data.numpy().shape,zR.shape)
			zH_L = zH.data.numpy().tolist()
			zR_L = zR.data.numpy().tolist()
			# dxdx = np.array(ddx)
			# print(dxdx.shape)
			ld[k][uid][l] = [zH_L,zR_L]
		# Mean errors
		ed[k][uid]['mean'] = [np.mean(np.asarray(e_H)),np.mean(np.asarray(e_R))]
		# Sum of harmonic and residual MSE's
		ed[k][uid]['sum'] = np.mean(np.asarray(e_H)) + np.mean(np.asarray(e_R))

# Saving the dict obtained
name_dict = './EDict.json'
with open(name_dict, 'w') as fp:
	json.dump(ed, fp)

# Saving the Latent sdict obtained
name_dict = './LSDict.json'
with open(name_dict, 'w') as fp:
	json.dump(ld, fp)



