import json
import numpy as np
import matplotlib.pyplot as pyp
import csv

# Saving the dict obtained
name_dict = './EDict.json'
with open(name_dict) as json_file: 
    data = json.load(json_file) 

# pyp.figure()
# pyp.title('All in One')
# for k in data.keys():
# 	list_H = []
# 	list_R = []
# 	for kp in data[k].keys():
# 		print(kp)
# 		le_h = []
# 		le_r = []
# 		for n in data[k][kp].keys():
# 			print(n)
# 			le_h.append(data[k][kp][n][0])
# 			le_r.append(data[k][kp][n][1])
# 		list_notes = list(data[k][kp].keys())
# 		pyp.plot(list_notes,le_h,label = k + '_' + kp)
# 		# pyp.plot(list_notes,le_h,label = kp)
# 		pyp.legend(loc = 'best')
# pyp.savefig('./plots/' + 'AIO' + '.png')

list_notes = ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
csv_dir = './dump-csv/'

# Writing per note MSE's
for k in data.keys():
	print('Algo: ',k)
	name_af = k
	l_uid = ['n']
	for kp in data[k].keys():
		l_uid.append(kp[-6:])
		# print('First Row:',l_uid)
	# Writing entries into the .csv file
	with open(csv_dir + name_af + '_H.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(l_uid)
		for l in list_notes:
			dl_H = [l]
			for kp in data[k].keys():
				dl_H.append(data[k][kp][l][0])
			csvwriter.writerow(dl_H)

	with open(csv_dir + name_af + '_R.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(l_uid)
		for l in list_notes:
			dl_H = [l]
			for kp in data[k].keys():
				dl_H.append(data[k][kp][l][1])
			csvwriter.writerow(dl_H)

	with open(csv_dir + name_af + '_HpR.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(l_uid)
		for l in list_notes:
			dl_H = [l]
			for kp in data[k].keys():
				dl_H.append(data[k][kp][l][0] + data[k][kp][l][1])
			csvwriter.writerow(dl_H)


# Writing mean MSE across all notes and sum of MSE
for k in data.keys():
	print('Algo: ',k)
	name_af = k
	l_uid = ['n']
	for kp in data[k].keys():
		l_uid.append(kp[-6:])

	with open(csv_dir + name_af + '_meanH.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(l_uid)
		dl_H = ['n']
		for kp in data[k].keys():
			dl_H.append(data[k][kp]['mean'][0])
		csvwriter.writerow(dl_H)

	with open(csv_dir + name_af + '_meanR.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(l_uid)
		dl_H = ['n']
		for kp in data[k].keys():
			dl_H.append(data[k][kp]['mean'][1])
		csvwriter.writerow(dl_H)

	with open(csv_dir + name_af + '_sumHR.csv', "w") as csvfile:
		csvwriter = csv.writer(csvfile,  delimiter=',')
		csvwriter.writerow(l_uid)
		dl_H = ['n']
		for kp in data[k].keys():
			dl_H.append(data[k][kp]['sum'])
		csvwriter.writerow(dl_H)


