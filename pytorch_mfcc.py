#!/usr/bin/python
#-*- coding: utf-8 -*-
# This code is from https://github.com/skaws2003/pytorch-mfcc (MIT License)

import torch
import decimal
import numpy
from torch.autograd import Function
import math

def dct(x, norm=None):
    """
    ##This code fragment is from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py ##
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * numpy.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= numpy.sqrt(N) * 2
        V[:, 1:] /= numpy.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def hz2mel(hz):
    """
    Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1+hz/700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)


class MFCC(torch.nn.Module):
    def __init__(self,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True):
        super(MFCC,self).__init__()
        self.samplerate = samplerate
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.nfft = nfft or self.calculate_nfft()
        self.lowfreq = lowfreq
        self.highfreq = highfreq or self.samplerate/2
        self.preemph = preemph
        self.ceplifter = ceplifter
        self.appendEnergy = appendEnergy
        self.winfunc=lambda x:numpy.ones((x,))


    def calculate_nfft(self):
        """
        Calculates the FFT size as a power of two greater than or equal to
        the number of samples in a single window length.
        
        Having an FFT less than the window length loses precision by dropping
        many of the samples; a longer FFT than the window allows zero-padding
        of the FFT buffer which is neutral in terms of frequency domain conversion.
        :param samplerate: The sample rate of the signal we are working with, in Hz.
        :param winlen: The length of the analysis window in seconds.
        """
        window_length_samples = self.winlen * self.samplerate
        nfft = 1
        while nfft < window_length_samples:
            nfft *= 2
        return nfft

    
    def forward(self,signals,lengths):
        """
        Calculates MFCC.
        :param signals: (torch.Tensor) batch of signals padded by 0.
        :param lengths: (list) length of each elements in batch.
        """
        self.tensor_type = signals.dtype
        self.torch_device = signals.device
        outs = []
        for i,signal in enumerate(signals):
            feat,energy = self.fbank(signal[:lengths[i]])
            feat = torch.log(feat)
            feat = dct(feat,norm='ortho')[:,:self.numcep]
            feat = self.lifter(feat)
            if self.appendEnergy:
                feat[:,0] = torch.log(energy) # replace first cepstral coefficient with log of frame energy
            outs.append(feat)
        
        # Pad each element of outs list
        max_len = max(outs,key=lambda x: x.shape[0]).shape[0]
        mfcc_lengths = []
        for i in range(len(outs)):
            mfcc_lengths.append(len(outs[i]))
            zeros = torch.zeros((max_len-outs[i].shape[0],outs[i].shape[1]),dtype=self.tensor_type).to(self.torch_device)
            outs[i] = torch.cat([outs[i],zeros],dim=0)
        outs = torch.stack(outs)
        return outs,mfcc_lengths


    def fbank(self,signal):
        """
        Compute Mel-filterbank energy features from an audio signal.
        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The second return value is the energy in each frame (total energy, unwindowed)
        """
        signal = self.preemphasis(signal)
        frames = self.framesig(signal)
        pspec = self.powspec(frames)
        energy = torch.sum(pspec,dim=1) # this stores the total energy in each frame
        energy = energy + numpy.finfo(numpy.float32).eps # if energy is zero, we get problems with log

        fb = self.get_filterbanks()
        feat = torch.mm(pspec,fb) # compute the filterbank energies
        feat = feat + numpy.finfo(numpy.float32).eps # if feat is zero, we get problems with log

        return feat,energy

    
    def preemphasis(self,signal,coeff=0.95):
        """
        perform preemphasis on the input signal.
        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
        :returns: the filtered signal.
        """
        a = signal[0].view(1)
        b = signal[1:] - self.preemph * signal[:-1]
        return torch.cat([a,b])


    def framesig(self,signal):
        """
        Frame a signal into overlapping frames.
        :param sig: the audio signal to frame.
        :returns: an array of frames. Size is NUMFRAMES by frame_len.
        """
        frame_len = self.winlen * self.samplerate
        frame_step = self.winstep * self.samplerate

        slen = len(signal)
        frame_len = int(round_half_up(frame_len))
        frame_step = int(round_half_up(frame_step))
        if slen <= frame_len:
            numframes = 1
        else:
            numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

        padlen = int((numframes - 1) * frame_step + frame_len)

        zeros = torch.zeros((padlen-slen)).to(self.torch_device)

        padsignal = torch.cat((signal,zeros))
        
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        ind_shape = indices.shape
        indices = numpy.array(indices, dtype=numpy.int32).reshape([-1])
        frames = padsignal[indices].view(ind_shape)
        win = numpy.tile(self.winfunc(frame_len), (numframes, 1))
        win = torch.tensor(win,dtype=self.tensor_type).to(self.torch_device)

        return frames * win

    def powspec(self,frames):
        """
        Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
        :param frames: the array of frames. Each row is a frame.
        :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
        """
        maged = self.magspec(frames)
        return 1.0 / self.nfft * torch.mul(maged,maged)


    def magspec(self,frames):
        """
        Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
        :param frames: the array of frames. Each row is a frame.
        :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
        """
        if frames.shape[1] < self.nfft:
            fshape = frames.shape
            cat_zeros = torch.zeros([fshape[0],self.nfft-fshape[1]],dtype=self.tensor_type,device=self.torch_device)
            frames = torch.cat([frames,cat_zeros],dim=1)
        complex_spec = torch.rfft(frames,1)
        abs_spec = torch.sqrt(torch.sum(torch.mul(complex_spec,complex_spec),dim=2)) # complex absolute
        return abs_spec


    def get_filterbanks(self):
        """
        Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        
        # compute points evenly spaced in mels
        lowmel = hz2mel(self.lowfreq)
        highmel = hz2mel(self.highfreq)
        melpoints = numpy.linspace(lowmel,highmel,self.nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = numpy.floor((self.nfft+1)*mel2hz(melpoints)/self.samplerate)

        fbank = numpy.zeros([self.nfilt,self.nfft//2+1])
        for j in range(0,self.nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        rtn = torch.tensor(fbank.T,dtype=self.tensor_type,device=self.torch_device)
        return rtn


    def lifter(self,cepstra):
        """
        Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
        magnitude of the high frequency DCT coeffs.
        :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        feat,ceplifter
        """
        if self.ceplifter > 0:
            nframes,ncoeff = cepstra.shape
            n = torch.arange(ncoeff).type(self.tensor_type).to(self.torch_device)
            lift = 1 + (self.ceplifter/2.)*torch.sin(numpy.pi*n/self.ceplifter)
            return lift*cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra