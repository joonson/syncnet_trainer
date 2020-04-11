#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import cv2
import math
from scipy.io import wavfile

def loadWAV(filename, max_frames, start_frame=0, evalmode=False, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240
    start_audio = start_frame * 160

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        raise ValueError('Audio clip is too short');

    if evalmode:
        start_frame = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        start_frame = numpy.array([start_audio])
    
    feats = []
    for asf in start_frame:
        feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0)

    feat = torch.FloatTensor(feat)

    return feat;


def get_frames(filename, max_frames=100, start_frame=0):

    cap = cv2.VideoCapture(filename)

    cap.set(1,start_frame)

    images = []
    for frame_num in range(0,max_frames):
        ret, image = cap.read()
        images.append(image)

    cap.release()

    im = numpy.stack(images,axis=3)
    im = numpy.expand_dims(im,axis=0)
    im = numpy.transpose(im,(0,3,4,1,2))

    imtv = torch.FloatTensor(im)

    return imtv