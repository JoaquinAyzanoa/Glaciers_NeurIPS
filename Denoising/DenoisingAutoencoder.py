import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import torch.nn.functional as F  # Add this import at the top
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import * 
from datasetClasses import *
from DeepLearning.LSTMTrain_temperature3.functions import saveCheckpoint, loadCheckpoint

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Input: [batch_size, 1, 50, 50]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1),  # [batch, 32, 50, 50]
            nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),  # [batch, 32, 25, 25]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # [batch, 16, 25, 25]
            nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True)  # [batch, 16, 13, 13]
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8 )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # [batch, 32, 26, 26]
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),  # [batch, 32, 26, 26]
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=2),  # [batch, 1, 52, 52]
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 2:  # Input is (50, 50)
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif len(x.shape) == 3:  # Input is (batch, 50, 50)
            x = x.unsqueeze(1)  # Add channel dim
        else:
            # Assume input is already [batch, channels, height, width]
            pass

        #x = F.pad(x, (1, 1, 1, 1))  # Pad (left, right, top, bottom) to make 52x52
        x = self.encoder(x)
        x = self.decoder(x)
        #x = x[:, :, :50, :50]  # Crop back to 50x50
        return x.squeeze(1)  # Remove channel dim to match input shape


def trainLoop(trainLoader, valLoader, model, optimizer, criterion, loadModel, modelName, params, device, pathOrigin):
    
    # variables
    torch.autograd.set_detect_anomaly(True)
    trainLosses = np.ones(len(trainLoader) * params["epochs"])
    validationLosses = np.ones(len(trainLoader) * params["epochs"])
    trainCounter = 0
    valLoss = torch.zeros(1)
    val_loss = 0


    # load model and optimizer from checkpoint
    if loadModel:
        # get into folder
        os.chdir(pathOrigin + "/models")
        lastState = loadCheckpoint(model, optimizer, pathOrigin + "/models/" + modelName)
        model = lastState[0]
        optimizer = lastState[1]

    ###################### start training #############################

    for b in range(params["epochs"]):
        for inpts, targets in trainLoader:
            
            # use tokenizer on gpu
            model.train()
            # Ensure proper shape for model
            if len(inpts.shape) == 3:
                inpts = inpts.unsqueeze(1)
            inpts = inpts.to(device).float()
            targets = targets.to(device).float()
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            forward = model.forward(inpts)
            loss = criterion(forward, targets)  # add reconstruction loss
            loss.backward()
            optimizer.step()
            #trainCounter += 1
            
            # save loss
            with torch.no_grad():
                if trainCounter % params["validationStep"] == 0 and trainCounter != 0:
                    model.eval()
                    val_loss = 0
                    len_loss = 10
                    for i in range(len_loss):
                        x, y = next(iter(valLoader))
                        x = x.to(device).float()
                        y = y.to(device).float()

                        # Ensure target has the same shape as output
                        if len(y.shape) == 3:
                            y = y.unsqueeze(1)
                        
                        # predict
                        pred = model.forward(x)
                        loss_val = criterion(pred, y)
                        val_loss += loss_val.detach().cpu().item()
                    val_loss = val_loss / len_loss
                    print("validation loss: ", val_loss)

                #save for csv
                trainLosses[trainCounter] = loss.detach().cpu().item()
                validationLosses[trainCounter] = val_loss
            trainCounter += 1
            # save model and optimizer checkpoint in case of memory overlow
            if trainCounter % 500 == 0:
                saveCheckpoint(model, optimizer, pathOrigin + "/" + "models/" + modelName)

            print("epoch: ", b, ", example: ", trainCounter, " current loss = ", loss.detach().cpu().item())

    ## save model state
    saveCheckpoint(model, optimizer, pathOrigin + "/" + "models/" + modelName)
    print("results saved!")
    return


if __name__ == "__main__":
    pathOrigin = "C:\\FAU\\3rd SEM\\Glaciers_NeurIPS"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelName = "denoising_autoencoder_v2"
    # define hyperparameters
    params = {"learningRate": 0.0001, "weightDecay": 0.001, "epochs": 20, "batchSize": 128, "validationStep": 100}

    # get dataLoaders
    path_images = os.path.join(path, "datasets", name, "Denoising")

    #datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
    datasetTrain = simplexnoiseLoader(path_images, "train")
    dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

    datasetVal = simplexnoiseLoader(path_images, "val")
    #datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
    dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)

    

    model = DenoisingAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                lr = params["learningRate"], weight_decay= params["weightDecay"])
    loss = torch.nn.MSELoss()

    #loss = torch.nn.L1Loss()
    print('params', params)
    #implement training loop here
    trainLoop(dataTrain, dataVal, model, optimizer, loss, True, modelName, params, device, pathOrigin)


