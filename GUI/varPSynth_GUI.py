# GUI frame for the hprModel_function.py

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
    import tkFileDialog, tkMessageBox
except ImportError:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox
import sys, os
from scipy.io.wavfile import read
import varPSynth_function
sys.path.append('../Dependencies/')
sys.path.append('../Dependencies/models/')

import utilFunctions as UF
from essentia.standard import MonoLoader, PitchYinProbabilistic
import numpy as np
import harmonicModel as HM
from scipy.signal import windows
import matplotlib.pyplot as pyp
 
class pitch_synth:
  
	def __init__(self, parent):  
		 
		self.parent = parent        
		self.initUI()

	def initUI(self):

		# choose_label = "Input file (.wav, mono and 44100 sampling rate):"
		# Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
		# #TEXTBOX TO PRINT PATH OF THE SOUND FILE
		# self.filelocation = Entry(self.parent)
		# self.filelocation.focus_set()
		# self.filelocation["width"] = 25
		# self.filelocation.grid(row=1,column=0, sticky=W, padx=10)
		# self.filelocation.delete(0, END)
		# self.filelocation.insert(0, '../../sounds/sax-phrase-short.wav')

		# #BUTTON TO BROWSE SOUND FILE
		# self.open_file = Button(self.parent, text="Browse...", command=self.browse_file) #see: def browse_file(self)
		# self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 
		# #BUTTON TO PREVIEW SOUND FILE
		# self.preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()), bg="gray30", fg="white")
		# self.preview.grid(row=1, column=0, sticky=W, padx=(306,6))

		# Harmonic .pth file loading-------------------------------------------------------------------
		choose_label = ".pth file for the Harmonic Component"
		Label(self.parent, text=choose_label).grid(row=1, column=0, sticky=W, padx=5, pady=(10,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.pH = Entry(self.parent)
		self.pH.focus_set()
		self.pH["width"] = 25
		self.pH.grid(row=2,column=0, sticky=W, padx=10)
		self.pH.delete(0, END)
		self.pH.insert(0, '/home/krishna/Desktop/IITB/DDP/Main/ddp-genmodels/pitch_gen/Carnatic/ISMIR_expts/paramshpr/synthmanifold(2020-06-11 16:52:53.264191)_[253, True, [253, 60], 32, [32, 60, 253], 1]_niters_2000_lr_0.001_beta_0.1_netH.pth')

		# Choice of Synth
		choice_genmethod = "Choice of Synthesis/Generation"
		Label(self.parent, text=choice_genmethod).grid(row=3, column=0, sticky=W, padx=0, pady=(10,2))
		self.choice = StringVar()
		self.choice.set("VibratoP") # initial value
		window_option = OptionMenu(self.parent, self.choice,"VibratoP", "FileP")
		window_option.grid(row=3, column=0, sticky=W, padx=(220, 6), pady=(10,2))

		# Select the file from which to extract pitch
		choose_pitch = "Input Pitch File (.wav, mono and 44100 sampling rate):"
		Label(self.parent, text=choose_pitch).grid(row=4, column=0, sticky=W, padx=5, pady=(10,2))
		self.pitchfile = Entry(self.parent)
		self.pitchfile.focus_set()
		self.pitchfile["width"] = 25
		self.pitchfile.grid(row=5,column=0, sticky=W, padx=10)
		self.pitchfile.delete(0, END)
		self.pitchfile.insert(0, '/home/krishna/Desktop/QueenBRVocals.wav')

		#BUTTON TO BROWSE SOUND FILE
		self.open_file = Button(self.parent, text="Browse...", command=self.browse_pitch) #see: def browse_file(self)
		self.open_file.grid(row=5, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 
		#BUTTON TO PREVIEW SOUND FILE
		self.preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.pitchfile.get()), bg="gray30", fg="white")
		self.preview.grid(row=5, column=0, sticky=W, padx=(306,6))

		self.preview = Button(self.parent, text="PYin Pitch", command=self.play_pyin, bg="gray30", fg="white")
		self.preview.grid(row=5, column=0, sticky=W, padx=(340,15))

		# Note Duration
		note_dur_label = "Duration of Note (s):"
		Label(self.parent, text=note_dur_label).grid(row=6, column=0, sticky=W, padx=5, pady=(10,2))
		self.nD = Entry(self.parent, justify=CENTER)
		self.nD["width"] = 5
		self.nD.grid(row=6,column=0, sticky=W, padx=(180,5), pady=(10,2))
		self.nD.delete(0, END)
		self.nD.insert(0, "5")

		# Vibrato Center Frequency
		start_freq_label = "Vibrato Centre frequency (Hz):"
		Label(self.parent, text=start_freq_label).grid(row=7, column=0, sticky=W, padx=5, pady=(10,2))
		self.Vcf = Entry(self.parent, justify=CENTER)
		self.Vcf["width"] = 5
		self.Vcf.grid(row=7,column=0, sticky=W, padx=(180,5), pady=(10,2))
		self.Vcf.delete(0, END)
		self.Vcf.insert(0, "400")

		# Vibrato Depth as percent of Center Frequency
		end_freq_label = "Vibrato Depth (% of fc):"
		Label(self.parent, text=end_freq_label).grid(row=8, column=0, sticky=W, padx=5, pady=(10,2))
		self.Av = Entry(self.parent, justify=CENTER)
		self.Av["width"] = 5
		self.Av.grid(row=8,column=0, sticky=W, padx=(180,5), pady=(10,2))
		self.Av.delete(0, END)
		self.Av.insert(0, "1")

		# Vibrato Frequencu
		end_freq_label = "Vibrato Frequency (Hz):"
		Label(self.parent, text=end_freq_label).grid(row=9, column=0, sticky=W, padx=5, pady=(10,2))
		self.Vf = Entry(self.parent, justify=CENTER)
		self.Vf["width"] = 5
		self.Vf.grid(row=9,column=0, sticky=W, padx=(180,5), pady=(10,2))
		self.Vf.delete(0, END)
		self.Vf.insert(0, "5")

		#BUTTON TO COMPUTE EVERYTHING
		self.compute = Button(self.parent, text="Compute", command=self.compute_model, bg="dark red", fg="white")
		self.compute.grid(row=10, column=0, padx=5, pady=(10,2), sticky=W)

		#BUTTON TO PLAY SINE OUTPUT
		dir_output = './audioout/'
		output_label = "Network Generated Audio:"
		Label(self.parent, text=output_label).grid(row=11, column=0, sticky=W, padx=5, pady=(10,0))
		# if(str(self.choice.get()) == 'DrawP'):
			# self.output = Button(self.parent, text=">", command=lambda:UF.wavplay(dir_output + str(self.pitchfile.get()) + '_genH.wav'), bg="gray30", fg="white")
		# else:
		self.output = Button(self.parent, text=">", command=lambda:UF.wavplay(dir_output + os.path.basename(str(self.pitchfile.get()))[:-4] + '_genH.wav'), bg="gray30", fg="white")
		self.output.grid(row=11, column=0, padx=(180,5), pady=(10,0), sticky=W)

		# #BUTTON TO PLAY STOCHASTIC OUTPUT
		# output_label = "Stochastic:"
		# Label(self.parent, text=output_label).grid(row=16, column=0, sticky=W, padx=5, pady=(5,0))
		# self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel_stochastic.wav'), bg="gray30", fg="white")
		# self.output.grid(row=16, column=0, padx=(80,5), pady=(5,0), sticky=W)

		# #BUTTON TO PLAY OUTPUT
		# output_label = "Output:"
		# Label(self.parent, text=output_label).grid(row=17, column=0, sticky=W, padx=5, pady=(5,15))
		# self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel.wav'), bg="gray30", fg="white")
		# self.output.grid(row=17, column=0, padx=(80,5), pady=(5,15), sticky=W)


		# define options for opening file
		self.file_opt = options = {}
		options['defaultextension'] = '.wav'
		options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
		options['initialdir'] = '../../sounds/'
		options['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'
 
	def browse_file(self):
		
		self.filename = tkFileDialog.askopenfilename(**self.file_opt)
 
		#set the text of the self.filelocation
		self.filelocation.delete(0, END)
		self.filelocation.insert(0,self.filename)

	def browse_pitch(self):
		
		self.filename = tkFileDialog.askopenfilename(**self.file_opt)
 
		#set the text of the self.filelocation
		self.pitchfile.delete(0, END)
		self.pitchfile.insert(0,self.filename)

	def compute_model(self):
		
		try:
			pH = str(self.pH.get())
			choice = str(self.choice.get())
			pitchF = str(self.pitchfile.get())
			nD = float(self.nD.get())
			Vcf = float(self.Vcf.get())
			Av = float(self.Av.get())
			Vf = float(self.Vf.get())

			varPSynth_function.main(pH,choice,nD,Vcf,Av,Vf,pitchF)

		except ValueError as errorMessage:
			tkMessageBox.showerror("Input values error", errorMessage)

	def play_pyin(self):
		W,H,N = 1024,256,2048
		fs = 44100
		dir_save = './audioout/'
		aud_load = MonoLoader(sampleRate = fs,filename = self.pitchfile.get())()
		# pyin = PitchYinProbabilistic(frameSize = N,hopSize = H,lowRMSThreshold = 0.0001,preciseTime = True)
		# pyin_p,vp = pyin(aud_load)

		F,M,P = HM.harmonicModelAnal(x = aud_load, fs = fs, w = windows.hann(W), N = N, H = H, t = -120, nH=  150,minSineDur = 0.02, minf0 = 140, maxf0 = 1300, f0et = 5, harmDevSlope = 0.01)
		hC = 2
		pyin_p = F[:,hC - 1]/hC

		# pyp.figure()
		# pyp.plot(pyin_p,label = 'TwM')
		# pyp.plot(pyin_p_pyin,label = 'PYin')
		# pyp.legend(loc = 'best')
		# pyp.show()

		x = np.array([])
		phase = 0.0
		for frame_no in range(len(pyin_p)):
			tone = 1.0 * np.cos((2*np.pi*pyin_p[frame_no]*np.arange(H)/fs) + phase)
			x = np.append(x, tone)
			phase = phase + 2*np.pi*pyin_p[frame_no]*(H)/fs

		UF.wavwrite(x,fs,dir_save + self.pitchfile.get().split('/')[-1][:-4] + '_pyin.wav')
		UF.wavplay(dir_save + self.pitchfile.get().split('/')[-1][:-4] + '_pyin.wav')
