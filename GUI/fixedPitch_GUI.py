# GUI frame for the dftModel_function.py

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
    import tkFileDialog, tkMessageBox
except ImportError:
    # for Python3
    from tkinter import *  ## notice lowercase 't' in tkinter here
    from tkinter import filedialog as tkFileDialog
    from tkinter import messagebox as tkMessageBox
import sys, os
from scipy.io.wavfile import read
import fixedPitch_function
sys.path.append('../Dependencies/')
sys.path.append('../Dependencies/models/')
import utilFunctions as UF

class fP:
  
	def __init__(self, parent):  
		 
		self.parent = parent        
		self.initUI()

	def initUI(self):

		choose_label = "Input audio file (.wav, mono and 44100 sampling rate):"
		Label(self.parent, text=choose_label).grid(row=0, column=0, sticky=W, padx=5, pady=(10,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.filelocation = Entry(self.parent)
		self.filelocation.focus_set()
		self.filelocation["width"] = 25
		self.filelocation.grid(row=1,column=0, sticky=W, padx=10)
		self.filelocation.delete(0, END)
		self.filelocation.insert(0, '/home/krishna/Desktop/IITB/DDP/Datasets/Annotated Data/Behringer/IndividualNotes/146-Ri1_U_So_At.wav')

		#BUTTON TO BROWSE SOUND FILE
		self.open_file = Button(self.parent, text="Browse...", command=self.browse_file_audio) #see: def browse_file(self)
		self.open_file.grid(row=1, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 
		#BUTTON TO PREVIEW SOUND FILE
		self.preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()), bg="gray30", fg="white")
		self.preview.grid(row=1, column=0, sticky=W, padx=(306,6))


		# Harmonic .pth file loading-------------------------------------------------------------------
		choose_label = ".pth file for the Harmonic Component"
		Label(self.parent, text=choose_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.pH = Entry(self.parent)
		self.pH.focus_set()
		self.pH["width"] = 25
		self.pH.grid(row=3,column=0, sticky=W, padx=10)
		self.pH.delete(0, END)
		self.pH.insert(0, '/home/krishna/Desktop/IITB/DDP/Main/ddp-genmodels/pitch_gen/Carnatic/ISMIR_expts/paramshpr/synthmanifold(2020-05-15 02:38:56.040501)_[63, True, [63, 60], 32, [32, 60, 63], 1]_niters_2000_lr_0.001_beta_0.1_netH.pth')

		#BUTTON TO BROWSE SOUND FILE
		self.pH_open_file = Button(self.parent, text="Browse...", command=self.browse_file_H) #see: def browse_file(self)
		self.pH_open_file.grid(row=3, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 
		# #BUTTON TO PREVIEW SOUND FILE
		# self.pH_preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()), bg="gray30", fg="white")
		# self.pH_preview.grid(row=3, column=0, sticky=W, padx=(306,6))

		# Residual .pth file loading-------------------------------------------------------------------
		choose_label = ".pth file for the Residual Component"
		Label(self.parent, text=choose_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10,2))
 
		#TEXTBOX TO PRINT PATH OF THE SOUND FILE
		self.pR = Entry(self.parent)
		self.pR.focus_set()
		self.pR["width"] = 25
		self.pR.grid(row=5,column=0, sticky=W, padx=10)
		self.pR.delete(0, END)
		self.pR.insert(0, '/home/krishna/Desktop/IITB/DDP/Main/ddp-genmodels/pitch_gen/Carnatic/ISMIR_expts/paramshpr/synthmanifold(2020-05-15 02:38:56.040501)_[50, False, [50, 50], 32, [32, 50, 50], 0]_niters_2000_lr_0.001_beta_0.1_netR.pth')

		#BUTTON TO BROWSE SOUND FILE
		self.pR_open_file = Button(self.parent, text="Browse...", command=self.browse_file_R) #see: def browse_file(self)
		self.pR_open_file.grid(row=5, column=0, sticky=W, padx=(220, 6)) #put it beside the filelocation textbox
 
		# #BUTTON TO PREVIEW SOUND FILE
		# self.pR_preview = Button(self.parent, text=">", command=lambda:UF.wavplay(self.filelocation.get()), bg="gray30", fg="white")
		# self.pR_preview.grid(row=3, column=0, sticky=W, padx=(306,6))

		## DFT MODEL

		# #ANALYSIS WINDOW TYPE
		# wtype_label = "Window type:"
		# Label(self.parent, text=wtype_label).grid(row=2, column=0, sticky=W, padx=5, pady=(10,2))
		# self.w_type = StringVar()
		# self.w_type.set("blackman") # initial value
		# window_option = OptionMenu(self.parent, self.w_type, "rectangular", "hanning", "hamming", "blackman", "blackmanharris")
		# window_option.grid(row=2, column=0, sticky=W, padx=(95,5), pady=(10,2))

		# #WINDOW SIZE
		# M_label = "Window size (M):"
		# Label(self.parent, text=M_label).grid(row=3, column=0, sticky=W, padx=5, pady=(10,2))
		# self.M = Entry(self.parent, justify=CENTER)
		# self.M["width"] = 5
		# self.M.grid(row=3,column=0, sticky=W, padx=(115,5), pady=(10,2))
		# self.M.delete(0, END)
		# self.M.insert(0, "511")

		# #FFT SIZE
		# N_label = "FFT size (N) (power of two bigger than M):"
		# Label(self.parent, text=N_label).grid(row=4, column=0, sticky=W, padx=5, pady=(10,2))
		# self.N = Entry(self.parent, justify=CENTER)
		# self.N["width"] = 5
		# self.N.grid(row=4,column=0, sticky=W, padx=(270,5), pady=(10,2))
		# self.N.delete(0, END)
		# self.N.insert(0, "1024")

		# #TIME TO START ANALYSIS
		# time_label = "Time in sound (in seconds):"
		# Label(self.parent, text=time_label).grid(row=5, column=0, sticky=W, padx=5, pady=(10,2))
		# self.time = Entry(self.parent, justify=CENTER)
		# self.time["width"] = 5
		# self.time.grid(row=5, column=0, sticky=W, padx=(180,5), pady=(10,2))
		# self.time.delete(0, END)
		# self.time.insert(0, ".2")

		#BUTTON TO COMPUTE EVERYTHING
		self.compute = Button(self.parent, text="Compute", command=self.compute_model, bg="dark red", fg="white")
		self.compute.grid(row=6, column=0, padx=5, pady=(10,15), sticky=W)

		# define options for opening file
		self.file_opt = options = {}
		options['defaultextension'] = '.wav'
		options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
		options['initialdir'] = '../../sounds/'
		options['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'


		# -------------------------------------------------------------------------------------------------
		# Playing the output waveforms, along with the input HpS one
		dir_output = './audioout/'
		#BUTTON TO PLAY SINE OUTPUT
		output_label = "HpR version of input : "
		Label(self.parent, text=output_label).grid(row=7, column=0, sticky=W, padx=5, pady=(10,0))
		self.output = Button(self.parent, text=">", command=lambda:UF.wavplay(dir_output + os.path.basename(self.filelocation.get())[:-4] + '_og_HpR.wav'), bg="gray30", fg="white")
		self.output.grid(row=7, column=0, padx=(150,5), pady=(10,0), sticky=W)

		#BUTTON TO PLAY STOCHASTIC OUTPUT
		output_label = "Network reconstruction : "
		Label(self.parent, text=output_label).grid(row=8, column=0, sticky=W, padx=5, pady=(5,0))
		self.output = Button(self.parent, text=">", command=lambda:UF.wavplay(dir_output + os.path.basename(self.filelocation.get())[:-4] + '_recon_HpR.wav'), bg="gray30", fg="white")
		self.output.grid(row=8, column=0, padx=(150,5), pady=(5,0), sticky=W)

		# #BUTTON TO PLAY OUTPUT
		# output_label = "Output:"
		# Label(self.parent, text=output_label).grid(row=17, column=0, sticky=W, padx=5, pady=(5,15))
		# self.output = Button(self.parent, text=">", command=lambda:UF.wavplay('output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_hpsModel.wav'), bg="gray30", fg="white")
		# self.output.grid(row=17, column=0, padx=(80,5), pady=(5,15), sticky=W)
 
	def browse_file_audio(self):
		
		self.filename = tkFileDialog.askopenfilename(**self.file_opt)
 
		#set the text of the self.filelocation
		self.filelocation.delete(0, END)
		self.filelocation.insert(0,self.filename)

	def browse_file_H(self):
		
		self.filename = tkFileDialog.askopenfilename(**self.file_opt)
 
		#set the text of the self.filelocation
		self.pH.delete(0, END)
		self.pH.insert(0,self.filename)

	def browse_file_R(self):
		
		self.filename = tkFileDialog.askopenfilename(**self.file_opt)
 
		#set the text of the self.filelocation
		self.pR.delete(0, END)
		self.pR.insert(0,self.filename)

	def compute_model(self):
		
		try:
			inputFile = str(self.filelocation.get())
			pH = str(self.pH.get())
			pR = str(self.pR.get())
			# print(inputFile,pH,pR)
			# print(inputFile)
			# print(pH)
			# print(pR)
			# window = self.w_type.get()
			# M = int(self.M.get())
			# N = int(self.N.get())
			# time = float(self.time.get())
			
			fixedPitch_function.main(inputFile,pH,pR)

		except ValueError as errorMessage:
			tkMessageBox.showerror("Input values error",errorMessage)
			
