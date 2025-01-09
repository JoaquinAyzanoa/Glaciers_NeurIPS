import pandas as pd
import functions
import torch
import os
from torch.utils.data import DataLoader
import datasetClasses
import numpy as np
import matplotlib.pyplot as plt
#import transformerBase
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import * 
import pickle

from datasetClasses import glaciers
import lstmAttention3
## global variables for project
### change here to run on cluster ####
pathOrigin = path
#pathOrigin = "C:\\FAU\\3rd SEM\\Glaciers_NeurIPS"
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cuda"
tokenizer = False

if tokenizer:
    # load tokenizer and model
    Tokenizer = tokenizer.tokenizer()
    os.chdir(pathOrigin + "/models")
    Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizer")
    Tokenizer = Tokenizer.to(device)

modelName = "LSTMAttention3_model1"
#Create forlder to save prints
os.makedirs(os.path.join(pathOrigin, "prints",modelName), exist_ok=True)
#model = ConvLSTM.ConvLSTMPredictor([64, 64, 24, 24, 64, 24]).to(device)
model = lstmAttention3.LSTM(1,1, 2500, 2500, 0.1, 5,  device).to(device)
#model = LSTM.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)
#model = UNet(1,1).to(device)


# load weights to transformers
model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", modelName))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "Unet"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "ConvLSTM"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "LSTMEncDec"))
print("loading models finished")

# get dataLoaders
path_images = os.path.join(pathOrigin, "datasets", name, "alignedAveragedDataNDSIPatched")
path_temperatures= os.path.join(pathOrigin, "datasets", name, "TemperatureDataPatched")
# dataLoader /home/jonas/datasets/parbati
datasetTest = glaciers(path_images, path_temperatures, "test")
#datasetTest = datasetClasses.glaciers("/home/jonas/datasets/parbati", "test", bootstrap = True)
dataTest = DataLoader(datasetTest, 256,  shuffle = False)

prediction_path = os.path.join(pathOrigin, "datasets", name, "Predictions")
os.makedirs(prediction_path, exist_ok=True)

with torch.no_grad():
    # do 2000 bootstrap iterations
    losses = []
    nIterations = 1
    MSELoss = torch.nn.MSELoss()
    MAELoss = torch.nn.L1Loss()
    modelResults = np.zeros((nIterations, 2))

    for b in range(nIterations):
        MSElosses = torch.zeros(len(dataTest))
        MAElosses = torch.zeros(len(dataTest))
        counter = 0
        

        # check model performance on bootstrapped testset
        for inpts, targets , temperatures, idx in dataTest:

            inpts = inpts.to(device).float()
            targets = targets.to(device).float().squeeze()
            temperatures = temperatures.to(device).float()

            if tokenizer:
                # encode with tokenizer and put to gpu
                inpts = functions.tokenizerBatch(Tokenizer, inpts, "encoding", device)
                targets = functions.tokenizerBatch(Tokenizer, targets, "encoding", device)

            # predict
            model.eval()
            forward = model.forward(inpts, temperatures, targets, training = False)

            if tokenizer:
                forward = Tokenizer.decoder(forward)
                forward = functions.tokenizerBatch(Tokenizer, forward, "decoding", device)
                forward = torch.reshape(forward, (1, forward.size(0), 50, 50))

            
            for index, id in enumerate(idx):
                vector = forward[index,: ,:, :].cpu().numpy()
                # save data object on drive
                with open(os.path.join(prediction_path, str(id)), "wb") as fp:
                    pickle.dump(vector, fp)

            counter += 1

    print("Predicions model results finished")