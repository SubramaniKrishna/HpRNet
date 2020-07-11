# PyTorch Dataloader for the Carnatic Dataset
"""
- Provide as input the pickle file which is output from the parametric processing of the audio files
- Additional parameters:
	* num_frames - Number of frames of audio to consider
	* list_octave - one or more (input as list) from ['L','M','U']
	* list_notes - one or more (input as list) from ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
	* list_loudness - one or more (input as list) from ['So','Lo']
	* list_style - one or more (input as list) from ['At','Sm']
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class CarnaticDL(Dataset):
	def __init__(self, filename, num_frames, list_octave,list_notes,list_loudness,list_style):
		data = pickle.load(open(filename, 'rb'))
		names = list(data.keys())

		self.names = []
		self.labels = []
		self.pitches = []
		self.loudness = []
		self.cc_H = []
		self.cc_R = []

		for it,k in enumerate(names):
			# Extract metadata from the name of the file:
			uid = k.split('_')[0]
			octave = k.split('_')[1]
			note = k.split('_')[2]
			style = k.split('_')[3]

			temp_file = data[k]
			p = temp_file['pitch']
			loudness = temp_file['loudness']
			if(loudness == 'So'):
				lid = 0
			else:
				lid = 1
			cc_H = temp_file['cc']
			cc_R = temp_file['res']


			if(octave in list_octave and note in list_notes and loudness in list_loudness and style in list_style):
				if(cc_H.shape[0] > num_frames):
					num_iters = num_frames
				else:
					num_iters = cc_H.shape[0]
				fc = 0
				for f in range(num_iters):
					self.names.append(str(uid) + '_frame_'+str(fc + 1))
					fc = fc + 1
					self.labels.append(str(uid))
					self.pitches.append(p[f])
					self.loudness.append(lid)
					self.cc_H.append(cc_H[f,:])
					self.cc_R.append(cc_R[f,:])

	def __len__(self):
		return len(self.names)

	def __getitem__(self, idx):
		return self.labels[idx], torch.tensor(self.pitches[idx]), torch.tensor(self.loudness[idx]), torch.tensor(self.cc_H[idx]), torch.tensor(self.cc_R[idx])