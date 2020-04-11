#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchaudio
    
class SyncNetModel(nn.Module):
    def __init__(self, nOut = 1024, stride = 1):
        super(SyncNetModel, self).__init__();

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5,7), stride=(1,1), padding=(2,2)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2)),

            nn.Conv2d(96, 256, kernel_size=(5,5), stride=(2,1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,1)),

            nn.Conv2d(256, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(256, 512, kernel_size=(4,1), padding=(0,0), stride=(1,stride)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        );

        self.netfcaud = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, nOut, kernel_size=1),
        );

        self.netfclip = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, nOut, kernel_size=1),
        );

        self.netfcspk = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, nOut, kernel_size=1),
        );

        self.netfcface = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, nOut, kernel_size=1),
        );

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(stride,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        );

        self.instancenorm   = nn.InstanceNorm1d(40)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)


    def forward_vid(self, x):

        ## Image stream

        mid = self.netcnnlip(x); 
        mid = mid.view((mid.size()[0], mid.size()[1], -1)); # N x (ch x 24)

        out1  = self.netfclip(mid);
        out2  = self.netfcface(mid);

        return out1, out2

    def forward_aud(self, x):

        ## Audio stream

        x = self.torchfb(x)+1e-6
        x = self.instancenorm(x.log())
        x = x[:,:,1:-1].detach()
        
        mid = self.netcnnaud(x.unsqueeze(1)); # N x ch x 24 x M
        mid = mid.view((mid.size()[0], mid.size()[1], -1)); # N x (ch x 24)

        out1  = self.netfcaud(mid);
        out2  = self.netfcspk(mid);

        return out1, out2