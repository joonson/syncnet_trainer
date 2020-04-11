#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy
import sys
import time
import os
import argparse
import pdb
import glob
import torch
from SyncNetDist import SyncNet

from tuneThreshold import tuneThresholdfromScore
from sklearn import metrics
from DatasetLoader import DatasetLoader


parser = argparse.ArgumentParser(description = "TrainArgs");

## Data loader
parser.add_argument('--maxFrames', type=int, default=30, help='');
parser.add_argument('--nBatchSize', type=int, default=30, help='');
parser.add_argument('--nTrainPerEpoch', type=int, default=100000, help='');
parser.add_argument('--nTestPerEpoch',  type=int, default=10000, help='');
parser.add_argument('--nDataLoaderThread', type=int, default=4, help='');

## Training details
parser.add_argument('--model', type=str, default="", help='Model name');
parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs');
parser.add_argument('--temporal_stride', type=int, default=1, help='');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every epoch');

## Joint training params
parser.add_argument('--alphaC', type=float, default=1.0, help='Sync weight');
parser.add_argument('--alphaI', type=float, default=1.0, help='Identity weight');

## Load and save
parser.add_argument('--initial_model', type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',     type=str, default="./data/exp01", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list', type=str, default="data/dev.txt", help='');
parser.add_argument('--verify_list', type=str, default="data/test.txt", help='');

## Speaker recognition test
parser.add_argument('--test_list', type=str, default="voxceleb/test_list.txt", help='Evaluation list');
parser.add_argument('--test_path', type=str, default="voxceleb/voxceleb1", help='Absolute path to the test set');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

args = parser.parse_args();

# ==================== MAKE DIRECTORIES ====================

model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)

if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

# ==================== LOAD MODEL ====================

s = SyncNet(**vars(args));

# ==================== EVALUATE LIST ====================

it          = 1;

scorefile = open(result_save_path+"/scores.txt", "a+");

for items in vars(args):
    print(items, vars(args)[items]);
    scorefile.write('%s %s\n'%(items, vars(args)[items]));
scorefile.flush()

# ==================== LOAD MODEL PARAMS ====================

modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(0,it-1):
    clr = s.updateLearningRate(args.lr_decay) 

# ==================== EVAL ====================

if args.eval == True:
        
    sc, lab = s.evaluateFromListSave(args.test_list, print_interval=100, test_path=args.test_path)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    print('EER %2.4f'%result[1])

    quit();

# ==================== LOAD DATA LIST ====================

print('Reading data ...')

trainLoader = DatasetLoader(args.train_list,  nPerEpoch=args.nTrainPerEpoch, **vars(args))
valLoader   = DatasetLoader(args.verify_list, nPerEpoch=args.nTestPerEpoch, evalmode=True, **vars(args))

print('Reading done.')

# ==================== CHECK SPK ====================

clr = s.updateLearningRate(1)

while(1):   
    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Start Iteration");

    loss, trainacc  = s.train_network(trainLoader, evalmode=False, alpI=args.alphaI, alpC=args.alphaC);
    valloss, valacc = s.train_network(valLoader,   evalmode=True);

    print(time.strftime("%Y-%m-%d %H:%M:%S"), "%s: IT %d, LR %f, TACC %2.2f, TLOSS %f, VACC %2.2f, VLOSS %f\n"%(args.save_path, it, max(clr), trainacc, loss, valacc, valloss));
    scorefile.write("IT %d, LR %f, TACC %2.2f, TLOSS %f, VACC %2.2f, VLOSS %f\n"%(it, max(clr), trainacc, loss, valacc, valloss));
    scorefile.flush()

    # ==================== SAVE MODEL ====================

    clr = s.updateLearningRate(args.lr_decay) 

    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Saving model %d" % it)
    s.saveParameters(model_save_path+"/model%09d.model"%it);

    if it >= args.max_epoch:
        quit();

    it+=1;
    print("");

scorefile.close();





