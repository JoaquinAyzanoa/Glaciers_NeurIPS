import os
import sys
import matplotlib.pyplot as plt
import pickle
from opensimplex import OpenSimplex
import numpy as np
from config import *

#Read all file in the directory
path = r'C:\FAU\3rd SEM\Glaciers_NeurIPS\datasets\parvati\alignedAveragedDataNDSI'
path_noise = r'C:\FAU\3rd SEM\Glaciers_NeurIPS\datasets\parvati\simplexnoise'
path_noise_patched = r'C:\FAU\3rd SEM\Glaciers_NeurIPS\datasets\parvati\simplexnoiseDataPatched'
#read all files in the directory
files = os.listdir(path)
#create directory if not exist
os.makedirs(path_noise, exist_ok=True)
os.makedirs(path_noise_patched, exist_ok=True)

def noisy_image():
    for img in range(len(files)):
        img = str(img)
        with open(os.path.join(path, img), 'rb') as fp:
            data = pickle.load(fp)
        
        temp = OpenSimplex(seed=42)
        noise = np.zeros_like(data, dtype=float)
        rows, cols = noise.shape
        #chose the delta values randomly from 2 to 5
        deltax = np.random.randint(2, 5)
        deltay = np.random.randint(2, 5)

        for i in range(rows):
            for j in range(cols):
                noise[i, j] = temp.noise2(i / deltax, j / deltay)

        noisy_image = data + 0.2*noise

        #save noisy_image as pickle file
        with open(os.path.join(path_noise, img), 'wb') as fp:
            pickle.dump(noisy_image, fp)
        print(f'Noisy image {img} created successfully')
    return noisy_image

def patch_noisy_image():
    #join path path_noise_patched with "images" and "targets"
    path_images = os.path.join(path_noise_patched, "images")
    path_targets = os.path.join(path_noise_patched, "targets")
    os.makedirs(path_images, exist_ok=True)
    os.makedirs(path_targets, exist_ok=True)
    counter = 0
    for img in range(len(files)):
        img = str(img)
        with open(os.path.join(path, img), 'rb') as fp:
            data = pickle.load(fp)
        with open(os.path.join(path_noise, img), 'rb') as fp:
            noisy_data = pickle.load(fp)
        
        #creates patches using stride variable and patch size
        rows, cols = data.shape
        for i in range(0, rows - patchSize + 1, stride):
            for j in range(0, cols - patchSize + 1, stride):
                data_patch = data[i:i + patchSize, j:j + patchSize]
                noisy_patch = noisy_data[i:i + patchSize, j:j + patchSize]
                #save patch as pickle file
                with open(os.path.join(path_images, str(counter)), 'wb') as fp:
                    pickle.dump(noisy_patch, fp)
                with open(os.path.join(path_targets, str(counter)), 'wb') as fp:
                    pickle.dump(data_patch, fp)
                # #plot the noisy image and the original image
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(noisy_patch, cmap='gray')
                # ax[0].set_title('Noisy Image')
                # ax[1].imshow(data_patch, cmap='gray')
                # ax[1].set_title('Original Image')
                # plt.show()
                #print(data_patch.shape, noisy_patch.shape)

                counter += 1
        print(f'Noisy image {img} patched successfully')



if __name__ == '__main__':
    #noisy_image()
    patch_noisy_image()
        