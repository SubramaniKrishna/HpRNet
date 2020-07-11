import json
import numpy as np
import matplotlib.pyplot as pyp
import csv
from sklearn import manifold, datasets
import itertools
flatten = itertools.chain.from_iterable
# list(flatten(l))

# X, color = datasets.make_s_curve(100, random_state=0)

# print(X.shape,color.shape)
# print(color)

# Saving the dict obtained
name_dict = './LSDict.json'
with open(name_dict) as json_file: 
    data = json.load(json_file)

list_notes = ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
dir_data = './dump-ls/'
cids = [0,1,2,3,4,5,6,7,8,9,10,11]
dict_cmap = {k:list_notes[k] for k in cids}

alg_tsne = manifold.TSNE(n_components = 2, init = 'pca', random_state = 0)

for k in data.keys():
	print('Algo: ',k)
	name_af = k
	# l_uid = ['n']
	# for kp in data[k].keys():
		# l_uid.append(kp[-6:])
	
	# c = []
	for kp in data[k].keys():
		cou = 0
		col_l = []
		for it,l in enumerate(list_notes):
			# c.append(it)
			zH = np.array(data[k][kp][l][0])
			zR = np.array(data[k][kp][l][1])
			# col_l.append([l] * zH.shape[0])
			if(cou == 0):
				zT = zH
				zTr = zR
				c = it*np.ones(zH.shape[0])

			else:
				zT = np.vstack((zT,zH))
				zTr = np.vstack((zTr,zR))
				c = np.hstack((c,it*np.ones(zH.shape[0])))
			cou = cou + 1
		# col_l = list(flatten(col_l))
		# print(len(list(flatten(col_l))))
		# Perform T-SNE on the latent space	
		nchoose = 2000
		row_h = np.random.choice(zT.shape[0], nchoose, replace = False)
		zi_h = zT[row_h, :]
		c_h = c[row_h]
		# coll_r = np.array(col_l)[row_h]
		zH_reduced = alg_tsne.fit_transform(zi_h)
		fig, ax = pyp.subplots()
		# pyp.figure(k + '_H_' + kp)
		# pyp.scatter(zH_reduced[:, 0], zH_reduced[:, 1], c = c_h, cmap=pyp.cm.Spectral)
		scatter = ax.scatter(zH_reduced[:, 0], zH_reduced[:, 1], c = c_h, cmap=pyp.cm.Spectral)
		handles, labels = scatter.legend_elements()
		legend1 = pyp.legend(handles, list_notes, loc="best", title="Notes")
		ax.add_artist(legend1)
		pyp.savefig(dir_data + k + '_H_' + kp[-6:] + '.png')
		# print(zTr.shape)
		row_r = np.random.choice(zTr.shape[0], nchoose, replace = False)
		zi_r = zTr[row_r, :]
		c_r = c[row_r]
		zR_reduced = alg_tsne.fit_transform(zi_r)
		fig1, ax1 = pyp.subplots()
		# pyp.figure(k + '_R_' + kp)
		scatter1 = ax1.scatter(zR_reduced[:, 0], zR_reduced[:, 1], c = c_r, cmap=pyp.cm.Spectral)
		handles, labels = scatter1.legend_elements()
		legend2 = pyp.legend(handles, list_notes, loc="best", title="Notes")
		ax1.add_artist(legend2)
		pyp.savefig(dir_data + k + '_R_' + kp[-6:] + '.png')
		# print(zT.shape,c.shape)
			# print(zH.shape,zT.shape)

		# Storing CSV dumps
		with open(dir_data + k + '_H_' + kp[-6:] + '.csv', "w") as csvfile:
			csvwriter = csv.writer(csvfile,  delimiter=',')
			csvwriter.writerow(['x','y','c'])

			for i in range(zH_reduced.shape[0]):
				csvwriter.writerow([zH_reduced[i, 0],zH_reduced[i, 1],c_h[i]])
		with open(dir_data + k + '_R_' + kp[-6:] + '.csv', "w") as csvfile:
			csvwriter = csv.writer(csvfile,  delimiter=',')
			csvwriter.writerow(['x','y','c'])

			for i in range(zR_reduced.shape[0]):
				csvwriter.writerow([zR_reduced[i, 0],zR_reduced[i, 1],c_r[i]])

# for i in range(zH_reduced.shape[0]):
	