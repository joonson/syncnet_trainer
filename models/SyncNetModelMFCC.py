#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import pytorch_mfcc
    
class SyncNetModel(nn.Module):
    def __init__(self, nOut = 1024, stride=1):
        super(SyncNetModel, self).__init__();

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            
            nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0), stride=(1,stride)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
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

        self.mfcc_layer = pytorch_mfcc.MFCC(samplerate=16000)

    def forward_aud(self, x):

        x1,mfcc_lengths = self.mfcc_layer(x,[x.size()[1]]*x.size()[0])      # Do mfcc
        x1 = x1.unsqueeze(1).transpose(2,3).detach()
        
        mid = self.netcnnaud(x1); # N x ch x 24 x M
        mid = mid.view((mid.size()[0], mid.size()[1], -1)); # N x (ch x 24)

        out1  = self.netfcaud(mid);
        out2  = self.netfcspk(mid);

        return out1, out2;

    def forward_vid(self, x):

        mid = self.netcnnlip(x); 
        mid = mid.view((mid.size()[0], mid.size()[1], -1)); # N x (ch x 24)

        out1  = self.netfclip(mid);
        out2  = self.netfcface(mid);

        return out1, out2;