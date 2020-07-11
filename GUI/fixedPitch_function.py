# function to call the main analysis/synthesis functions in software/models/dftModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
import utilFunctions as UF
import dftModel as DFT
import stft as STFT

# --------------------------------------------------------------------------------------------
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
import morphing as m
import func_envs as fe
from vae_krishna import cVAE_synth
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
from essentia.standard import MonoLoader
from scipy.io.wavfile import write
import sampling_synth as ss
from scipy.signal import windows, resample
from scipy import interpolate
import pickle

# Import custom Carnatic Dataloader
from carnatic_DL import *
import HpR_modified as HPRM
from essentia.standard import MonoLoader, StartStopSilence, FrameGenerator, PitchYinProbabilistic, PitchYin
import essentia


def cc_calc_varP(audio_inp, params, params_ceps, params_R):
    """
    """

    fs = params['fs']
    W = params['W']
    N = params['N']
    H = params['H']
    t = params['t']
    maxnSines = params['maxnSines']
    thresh = params_ceps['thresh']
    # ceps_coeffs = params_ceps['ceps_coeffs']
    num_iters = params_ceps['num_iters']
    w = windows.hann(W)

    # Parameters for the residual cc calculation
    sf = params_R['sf']
    Fr = params_R['Fr']
    Hr = params_R['Hr']
    ceps_coeffs_residual = params_R['ceps_coeffs_residual']

    F, M, P = HPRM.harmonicModel_pyin(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH=  maxnSines, minSineDur = 0.02, harmDevSlope = 0.1)
    # F, M, P = HM.harmonicModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH=  maxnSines,minSineDur = 0.02, minf0 = 140, maxf0 = 1300, f0et = 5, harmDevSlope = 0.01)

    # Cepstral Coefficients Calculation
    CC_harm = np.zeros((F.shape[0], N))
    # CC_res = np.zeros((F.shape[0], Fr))   

    # Extract the pitch from returned F
    hC = 10
    pitch_pyin = F[:,hC - 1]/hC
    most_likely_pitch = pitch_pyin[pitch_pyin.shape[0]//2]
    pitch_pyin = most_likely_pitch*np.ones(pitch_pyin.shape)

    # Define the new magnitude array based on the modified coefficients sampled from the TAE
    new_M = np.zeros_like(M)
    new_F = np.zeros_like(F)
    new_P = np.zeros_like(P)

    for i in range(pitch_pyin.shape[0]):
        hm_arr = np.arange(maxnSines) + 1
        f0 = pitch_pyin[i]
        new_F[i,:] = f0*hm_arr 

    new_F = F
    # Compute the TAE over the Harmonic model

    for i in range(F.shape[0]):
        f0 = pitch_pyin[i]
        if(f0 != 0):
            ceps_coeffs = (int)(1.9*(fs/(2*f0)))
            # ceps_coeffs = 100
        else:
            ceps_coeffs = N

        # Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
        # f = interpolate.interp1d((F[i,:]/fs)*N,M[i,:]/20,kind = 'linear',fill_value = '-6', bounds_error=False)
        f = interpolate.interp1d(F[i,:],M[i,:]/20,kind = 'linear',fill_value = '-6', bounds_error=False)
        # Frequency bins
        fbins = np.linspace(0,fs/2,N//2)
        # fbins = np.arange(N)
        finp = f(fbins)
        zp = np.concatenate((finp[0:N//2],np.array([0]),np.flip(finp[1:N//2])))
        # print(zp.shape)
        specenv,_,_ = fe.calc_true_envelope_spectral(zp,N,thresh,ceps_coeffs,num_iters)
        CC_harm[i,:] = np.real(np.fft.ifft(specenv))
        fp = interpolate.interp1d(fbins,specenv[:N//2],kind = 'linear',fill_value = '-6', bounds_error=False)
        new_M[i,:] = 20*fp(new_F[i,:])


    # Compute the residual by subtracting the above from the original audio
    # Ns = N
    residual = UF.sineSubtraction(audio_inp,N,H,new_F,new_M,P,fs)

    # Compute the framewise TAE for the 'stochastic residual' as an approximation of the residual envelope
    cou = 0
    for frame in FrameGenerator(essentia.array(residual), frameSize = Fr, hopSize = Hr):
        # f0 = pitch_pyin[i]
        # Sub-sample the residual spectrum
        fft_res = np.log10(np.abs(np.fft.fft(frame,Fr)))
        fft_res_ss = resample(fft_res,int(sf*Fr))
        # Resample to original size
        fft_res_ss = resample(fft_res_ss, Fr)
        # ceps_coeffs_residual = ceps_coeffs_residual
        # mY,_,_,_ = fe.calc_true_envelope(frame*windows.hann(Fr),Fr,thresh,ceps_coeffs_residual,num_iters)
        mY,_,_ = fe.calc_true_envelope_spectral(fft_res_ss,Fr,thresh,ceps_coeffs_residual,num_iters)
        # mY = fft_res_ss
        # CC_res[cou,:] = np.real(np.fft.ifft(mY))
        # mY = 20*mY[:Fr//2 + 1]
        if(cou == 0):
            CC_res = np.array([np.real(np.fft.ifft(mY))])
            spec_res = np.array([fft_res])
            spec_res_ss = np.array([fft_res_ss])
        else:
            CC_res = np.vstack((CC_res, np.array([np.real(np.fft.ifft(mY))])))
            spec_res = np.vstack((spec_res,np.array([fft_res])))
            spec_res_ss = np.vstack((spec_res_ss,np.array([fft_res_ss])))

        cou = cou + 1

    return CC_harm, pitch_pyin, CC_res

######################################################################

# --------------------------------------------------------------------------------------------


def main(audFile,pH,pR):
    """
    inputFile: input sound file (monophonic with sampling rate of 44100)
    window: analysis window type (choice of rectangular, hanning, hamming, blackman, blackmanharris)
    M: analysis window size (odd integer value)
    N: fft size (power of two, bigger or equal than than M)
    time: time  to start analysis (in seconds)
    """

    # # read input sound (monophonic with sampling rate of 44100)
    # fs, x = UF.wavread(inputFile)

    # # compute analysis window
    # w = get_window(window, M)

    # # get a fragment of the input sound of size M
    # sample = int(time*fs)
    # if (sample+M >= x.size or sample < 0):                          # raise error if time outside of sound
    #     raise ValueError("Time outside sound boundaries")
    # x1 = x[sample:sample+M]

    # # compute the dft of the sound fragment
    # mX, pX = DFT.dftAnal(x1, w, N)

    # # compute the inverse dft of the spectrum
    # y = DFT.dftSynth(mX, pX, w.size)*sum(w)

    # # create figure
    # plt.figure(figsize=(12, 9))

    # # plot the sound fragment
    # plt.subplot(4,1,1)
    # plt.plot(time + np.arange(M)/float(fs), x1)
    # plt.axis([time, time + M/float(fs), min(x1), max(x1)])
    # plt.ylabel('amplitude')
    # plt.xlabel('time (sec)')
    # plt.title('input sound: x')

    # # plot the magnitude spectrum
    # plt.subplot(4,1,2)
    # plt.plot(float(fs)*np.arange(mX.size)/float(N), mX, 'r')
    # plt.axis([0, fs/2.0, min(mX), max(mX)])
    # plt.title ('magnitude spectrum: mX')
    # plt.ylabel('amplitude (dB)')
    # plt.xlabel('frequency (Hz)')
    # # plot the phase spectrum
    # plt.subplot(4,1,3)
    # plt.plot(float(fs)*np.arange(pX.size)/float(N), pX, 'c')
    # plt.axis([0, fs/2.0, min(pX), max(pX)])
    # plt.title ('phase spectrum: pX')
    # plt.ylabel('phase (radians)')
    # plt.xlabel('frequency (Hz)')

    # # plot the sound resulting from the inverse dft
    # plt.subplot(4,1,4)
    # plt.plot(time + np.arange(M)/float(fs), y)
    # plt.axis([time, time + M/float(fs), min(y), max(y)])
    # plt.ylabel('amplitude')
    # plt.xlabel('time (sec)')
    # plt.title('output sound: y')

    # plt.tight_layout()
    # plt.ion()
    # plt.show()

    # print('Entering the Main Function....')

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


    # Residual .pth file
    list_params = ast.literal_eval(pR.split('/')[-1].split('_')[1])
    file_load_RCVAE = pR
    rid = pR.split('/')[-1].split('_')[0][-7:-1]
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
    fs = Fs
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
    params_R['ceps_coeffs_residual'] = 50
    params_R['sf'] = 0.1

    # Pitch normalizing factor
    pnf = (int)(1.9*(44100/(2*dim_cc)))

    # Dimensionality of cepstral input to network (defined by lowest frequency, update accordingly!)
    cc_keep_H = (int)(1.9*(44100/(2*pnf)))
    cc_keep_R = 50

    # # Load the data
    # datafile = '../dataset_analysis/Carnatic_Processing_Dump_B_CHpS.txt'
    # data = pickle.load(open(datafile, 'rb'))
    # # Dataset conditions when testing
    # # ['L','M','U']
    # list_octave = ['U']
    # # ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2','Pa','Dha1','Dha2','Ni2','Ni3']
    # list_notes = ['Sa','Ri1','Ri2','Ga2','Ga3','Ma1','Ma2']
    # # ['So','Lo']
    # list_loudness = ['So','Lo']
    # # ['Sm','At']
    # list_style = ['Sm']

    # dir_save = './INet_' + hid + '_' + rid + '/'
    # try: 
    #     os.mkdir(dir_save) 
    # except OSError as error: 
    #     print('Dir Exists')

    # t = 0
    # for k in data.keys():
        # print (t+1,'/',len(data.keys()))
        # t = t + 1
        
        # temp_file = data[k]
        # p = temp_file['pitch']
        # cc_H = temp_file['cc']
        # cc_R = temp_file['res']

        # uid = k.split('_')[0]
        # octave = k.split('_')[1]
        # note = k.split('_')[2]
        # style = k.split('_')[3]
        # loudness = temp_file['loudness']

        # if(octave in list_octave and note in list_notes and loudness in list_loudness and style in list_style):
    dir_save = './audioout/'
    audio_ex = MonoLoader(sampleRate = Fs,filename = audFile)()
    cc_H, p, cc_R = cc_calc_varP(audio_ex,params,params_ceps,params_R)

    mX_og, pX_og = STFT.stftAnal(audio_ex, w, N, H)

    p = torch.FloatTensor(p)
    lenP = p.shape[0]
    inpH = torch.FloatTensor(cc_H[:lenP,:cc_keep_H])
    inpR = torch.FloatTensor(cc_R[:lenP,:cc_keep_R])
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

    mX_recon, pX_recon = STFT.stftAnal(hps_recon_cVAE, w, N, H)

    # Save the files as .wav files
    # write(filename = dir_save + '_' + str(k) + '_' + loudness + '_og_H.wav', rate = params['fs'], data = a_og.astype('float32'))
    # write(filename = dir_save + '_' + str(k) + '_' + loudness + '_og_R.wav', rate = params['fs'], data = sr_og.astype('float32'))
    
    write(filename = dir_save + audFile.split('/')[-1][:-4] + '_og_HpR.wav', rate = params['fs'], data = hps_og.astype('float32'))
    # write(filename = dir_save + '_' + str(k) + '_' + loudness + '_recon_H.wav', rate = params['fs'], data = a_recon_cVAE.astype('float32'))
    # write(filename = dir_save + '_' + str(k) + '_' + loudness + '_recon_R.wav', rate = params['fs'], data = sr_recon_cVAE.astype('float32'))
    write(filename = dir_save + audFile.split('/')[-1][:-4] + '_recon_HpR.wav', rate = params['fs'], data = hps_recon_cVAE.astype('float32'))

    # Plotting the spectrograms
    plt.figure('Spectrograms',figsize=(12, 9))
    # frequency range to plot
    maxplotfreq = 10000.0

    # plot original magnitude spectrogram
    plt.subplot(2,1,1)
    plt.title('Original magnitude spectrogram')
    numFrames = int(mX_og[:,0].size)
    frmTime = H*np.arange(numFrames)/float(Fs)
    binFreq = Fs*np.arange(N*maxplotfreq/fs)/N
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX_og[:,:int(N*maxplotfreq/fs+1)]))
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    # plt.title('magnitude spectrogram')
    plt.autoscale(tight=True)

    plt.subplot(2,1,2)
    plt.title('Reconstructed magnitude spectrogram')
    numFrames = int(mX_recon[:,0].size)
    frmTime = H*np.arange(numFrames)/float(Fs)
    binFreq = Fs*np.arange(N*maxplotfreq/fs)/N
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX_recon[:,:int(N*maxplotfreq/fs+1)]))
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    # plt.title('magnitude spectrogram')
    plt.autoscale(tight=True)

    plt.tight_layout()
    plt.ion()
    plt.show()


if __name__ == "__main__":
    main()
