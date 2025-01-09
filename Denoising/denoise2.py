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
import pickle
import matplotlib.pyplot as plt

pathOrigin = "C:\\FAU\\3rd SEM\\Glaciers_NeurIPS"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelName = "denoising_autoencoder_v2"
# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.001, "epochs": 20, "batchSize": 128, "validationStep": 100}

model = DenoisingAutoencoder().to(device)
path_model = os.path.join(pathOrigin, "prints",modelName)
os.makedirs(path_model, exist_ok=True)

# load weights to transformers
model = loadCheckpoint(model, None, os.path.join(pathOrigin, "models", modelName))

path_denoise = os.path.join(path, "datasets", name, "Predictions")

image = torch.zeros(256, 4, 50, 50).to(device)

target_path = r'C:\FAU\3rd SEM\Glaciers_NeurIPS\datasets\parvati\Test\alignedAveragedDataNDSIPatched\targets'

#print all files in the directory
for file in os.listdir(path_denoise):
    img = os.path.join(path_denoise, file)
    target = os.path.join(target_path, file)
    #read pickle file
    with open(img, 'rb') as f:
        prediction = pickle.load(f)
    with open(target, 'rb') as f:
        target = pickle.load(f)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.array(prediction[0 ,:, :]), cmap='gray')
    axs[0].set_title('Predicted')
    axs[1].imshow(np.array(target[0, :, :]), cmap='gray')
    axs[1].set_title('Target')
    os.chdir(os.path.join(path_model))
    plt.savefig(file+ ".jpeg")
    plt.close()
    
    #convert to tensor from (800,800) to (1,800,800)
    prediction = torch.tensor(prediction).to(device).float()
    prediction = prediction.unsqueeze(0)
    results = model.forward(prediction)
    image[int(file), :, :, :] = results.squeeze(0)


img_recovery = torch.zeros((4,800,800))
for i in range(16):
    for j in range(16):
        img_recovery[:,i*50:(i+1)*50,j*50:(j+1)*50] = image[j*16+i]

for i in range(4):
    img2save = img_recovery[i].cpu().detach().numpy()
    #save the denoised image as jpeg
    img2save = Image.fromarray(img2save*255)
    img2save = img2save.convert('L')
    img2save.save(os.path.join(path_model, "prediction_"+str(i)+".jpeg"))
    print(f'prediction_{str(i)} denoised successfully')

    