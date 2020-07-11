"""
Carnatic Model train cVAE only on the harmonic part
"""

# Dependencies
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as pyp
import numpy as np
from datetime import datetime

# Importing our model
import sys
sys.path.append('../Dependencies/')
sys.path.append('../Dependencies/models/')
from vae_krishna import cVAE_synth

# Import custom Carnatic Dataloader
from carnatic_DL import *

from time import time
import csv

# Defining the VAE loss function as mentioned in the original VAE paper(https://arxiv.org/pdf/1312.6114.pdf)
def loss_fn(recon_x, x, mean, var, beta = 1):
	MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
	KLD = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
	return ((MSE/x.size(0)) + beta*KLD), MSE, KLD


# Setting the device to cuda if GPU is available, else to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Load data into dataloader
batch_size = 512
datafile = '../dataset_analysis/Carnatic_Processing_Dump_B_CHpS.txt'

# ['L','M','U']
list_octave = ['U']
# ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
list_notes = ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2']
# list_notes = ['Sa','Ni3']
# ['So','Lo']
list_loudness = ['So','Lo']
# ['Sm','At']
list_style = ['At']

num_frames = 50000
dataset = CarnaticDL(filename = datafile, num_frames = num_frames, list_octave = list_octave, list_notes = list_notes, list_loudness = list_loudness, list_style = list_style)
# Split data into train and test
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
data_loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
len_train = len(train_dataset)
len_test = len(test_dataset)

# Starting the training
num_epochs = 2000

# Define the learning rate
learning_rate = 1.0e-3

# Pitch normalizing factor
pnf = 660

# Minimum number of CC's to keep, defined by the lowest pitch (and the multiplying factor, 1.9 in this case!)
# cc_keep = (int)((1.9)*(Fs/(2*165))) # If using all 3 octaves
cc_keep_H = (int)(1.9*(44100/(2*660)))
cc_keep_R = 50
# print(cc_keep)

for fc in [True]:
	for beta in [0.1]:
		for ld in [32]:
			# Defining the model architecture
			dim_cc_H = cc_keep_H
			dim_cc_R = cc_keep_R
			flag_cond = fc
			layer_dims_encH = [dim_cc_H,60]
			latent_dimsH = ld
			layer_dims_decH = [ld,60,dim_cc_H]
			layer_dims_encR = [dim_cc_R,50]
			latent_dimsR = ld
			layer_dims_decR = [ld,50,dim_cc_R]
			# layer_dims_dec.append(ld)
			# layer_dims_dec.reverse()
			# print(layer_dims_enc,layer_dims_dec)
			num_cond = 1

			now = datetime.now()
			dir_networkH = './paramshpr/synthmanifold(' + str(now) + ')_' + str([dim_cc_H, flag_cond, layer_dims_encH, latent_dimsH, layer_dims_decH, num_cond]) + '_niters_' + str(num_epochs) + '_lr_' + str(learning_rate) + '_beta_' + str(beta) + '_netH.pth'
			dir_networkR = './paramshpr/synthmanifold(' + str(now) + ')_' + str([dim_cc_R, False, layer_dims_encR, latent_dimsR, layer_dims_decR, 0]) + '_niters_' + str(num_epochs) + '_lr_' + str(learning_rate) + '_beta_' + str(beta) + '_netR.pth'
			
			# Define the model by instantiating an object of the class
			cVAE_H = cVAE_synth(flag_cond = flag_cond, layer_dims_enc = layer_dims_encH, layer_dims_dec = layer_dims_decH, latent_dims = latent_dimsH, num_cond = num_cond, device = device).to(device)
			cVAE_R = cVAE_synth(flag_cond = False, layer_dims_enc = layer_dims_encR, layer_dims_dec = layer_dims_decR, latent_dims = latent_dimsR, num_cond = 0, device = device).to(device)


			# Defining the optimizer
			optimizer_H = torch.optim.Adam(cVAE_H.parameters(), lr=learning_rate)
			optimizer_R = torch.optim.Adam(cVAE_R.parameters(), lr=learning_rate)

			# List to store loss
			loss_epoch = []
			MSE_loss = []
			kld_loss =[]
			test_loss = []
			# loss_avg = []
			start = time()

			for epoch in range(num_epochs):
				# bp = 1.0e-9*np.minimum((0.01*epoch),1)
				bp = beta

				tmpsum_H = 0
				tKLD_H = 0
				tmptrain_H = 0
				tmpsum_R = 0
				tKLD_R = 0
				tmptrain_R = 0
				it = 0
				for iteration, (label, pitch, loudness, x_H,x_R) in enumerate(data_loader_train):
					# print(x_H.shape,x_R.shape,cc_keep)
					temp_H = x_H[:,:cc_keep_H]
					temp_R = x_R[:,:cc_keep_R]
					xH = temp_H.contiguous().to(device, non_blocking=True)
					xR = temp_R.contiguous().to(device, non_blocking=True)
					# xH = temp_H.to(device, non_blocking=True)
					# xR = temp_R.to(device, non_blocking=True)
					# print(xH.shape,xR.shape,cc_keep)
					# Normalizing pitch to keep the range around 1
					f0 = (pitch.float()/pnf).to(device, non_blocking=True)
					it = it + 1

					c = f0.view(-1,1)
					# print(c.shape)
					# print(c)
					# print(layer_dims_enc,layer_dims_dec,xH.shape,c.shape)


					# With Pitch Conditioning
					if(flag_cond == True):
						xH_recon, muH, sigma_covarH, zH = cVAE_H(xH.float(),c.float())
						xR_recon, muR, sigma_covarR, zR = cVAE_R(xR.float(),c.float())
					# Without Pitch Conditioning
					else:
						xH_recon, muH, sigma_covarH, zH = cVAE_H(xH.float())
						xR_recon, muR, sigma_covarR, zR = cVAE_R(xR.float())

			        # Calculating the ELBO-Loss for Harmonic Network
					calcH = loss_fn(xH_recon.float(), xH.float(), muH, sigma_covarH, bp)
					lossH = calcH[0]
					MSE_H = calcH[1].item()
					KLD_H = calcH[2].item()
					tmptrain_H = tmptrain_H + lossH.item()
					tmpsum_H = tmpsum_H + MSE_H
					tKLD_H = tKLD_H + KLD_H

					# Harmonic
					optimizer_H.zero_grad()
					lossH.backward()
					optimizer_H.step()

					# Calculating the ELBO-Loss for Residual Network
					calcR = loss_fn(xR_recon.float(), xR.float(), muR, sigma_covarR, bp)
					lossR = calcR[0]
					MSE_R = calcR[1].item()
					KLD_R = calcR[2].item()
					tmptrain_R = tmptrain_R + lossR.item()
					tmpsum_R = tmpsum_R + MSE_R
					tKLD_R = tKLD_R + KLD_R
					# loss_vals.append(loss.item())

					# Residual
					optimizer_R.zero_grad()
					lossR.backward()
					optimizer_R.step()

					del lossH
					del calcH
					del MSE_H
					del KLD_H
					del lossR
					del calcR
					del MSE_R
					del KLD_R
				loss_epoch.append([tmptrain_H/len_train,tmptrain_R/len_train])
				MSE_loss.append([tmpsum_H/len_train,tmpsum_R/len_train])
				kld_loss.append([tKLD_H/len_train,tKLD_R/len_train])
				# MSE.append(MSE)
				# print(ld,bp,epoch,'loss = ',loss_epoch[-1],'MSE = ',MSE_loss[-1],'KLD = ',kld_loss[-1])
				print(fc,ld,bp,epoch,'harmonic loss = ',loss_epoch[-1][0],'residual loss = ',loss_epoch[-1][1])

			end = time()
			print('Device Used : ',device)
			print('Total time taken(minutes) : ',(end - start)*1.0/60)
			print('Time per epoch(seconds) :',(end - start)*1.0/num_epochs)


			torch.save(cVAE_H.state_dict(), dir_networkH)
			torch.save(cVAE_R.state_dict(), dir_networkR)

			descriptor_file = './paramshpr/descriptor_pthfiles.csv'

			params_write = [now, ld, num_frames, num_epochs, learning_rate, beta, list_octave, list_notes, list_style, list_loudness, datafile, flag_cond]

			with open(descriptor_file, 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(params_write)
			csvFile.close()