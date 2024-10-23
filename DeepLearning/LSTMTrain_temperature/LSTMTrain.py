import LSTM
import functions
import torch
import os
from torch.utils.data import DataLoader
import datasetClasses
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import *  

device = "cuda"
#device = "cpu"

# args: lstmLayers, lstmHiddenSize, lstmInputSize, dropout
#model = LSTM.LSTM(3,3, 2500, 2500, 0.1, device).to(device)
model = LSTM.LSTM(1,1, 2500, 2500, 0.1, device).to(device)

# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.001, "epochs": 40, "batchSize": 1, "optimizer": "adam", "validationStep": 100}


# get dataLoaders
path_images = os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched")
path_temperatures= os.path.join(path, "datasets", name, "TemperatureDataPatched")
#datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
datasetTrain = datasetClasses.glaciers(path_images, path_temperatures, "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers(path_images, path_temperatures, "val")
#datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)

# criterion
loss = torch.nn.MSELoss()

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoop(dataTrain, dataVal,  model, loss, False, "LSTMEncDec", params, True, device)




