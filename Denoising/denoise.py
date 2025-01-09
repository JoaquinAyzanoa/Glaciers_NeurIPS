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
from DenoisingAutoencoder import DenoisingAutoencoder
from PIL import Image

pathOrigin = "C:\\FAU\\3rd SEM\\Glaciers_NeurIPS"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelName = "denoising_autoencoder"
# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.001, "epochs": 20, "batchSize": 128, "validationStep": 100}

model = DenoisingAutoencoder().to(device)
path_model = os.path.join(pathOrigin, "prints",modelName)
os.makedirs(path_model, exist_ok=True)

# load weights to transformers
model = loadCheckpoint(model, None, os.path.join(pathOrigin, "models", modelName))

path_denoise = os.path.join(pathOrigin, "prints",'ToDenoise')

#print all files in the directory
for file in os.listdir(path_denoise):
    img = os.path.join(path_denoise, file)
    #read the grayscale image jpeg and convert to a tensor from 0 to 1
    img = Image.open(img).convert('L')
    img = np.array(img)
    img = img/255
    #convert to tensor from (800,800) to (1,800,800)
    img = torch.tensor(img).to(device).float()
    

    list_img = []
    for i in range(16):
        for j in range(16):
            list_img.append(img[i*50:(i+1)*50,j*50:(j+1)*50])
    img = torch.stack(list_img)
    
    img = img.unsqueeze(1)
    print(img.shape)
    results = model.forward(img)

    img_recovery = torch.zeros((800,800))
    for i in range(16):
        for j in range(16):
            img_recovery[i*50:(i+1)*50,j*50:(j+1)*50] = results[i*16+j]

    img_recovery = img_recovery.cpu().detach().numpy()
    #save the denoised image as jpeg
    img_recovery = Image.fromarray(img_recovery*255)
    img_recovery = img_recovery.convert('L')
    img_recovery.save(os.path.join(path_model, file))
    print(f'{file} denoised successfully')

    