import os
from torch.utils.data import Dataset
from torch import is_tensor
from torch.utils.data import DataLoader
import numpy as np
import pickle

def openData(name):
    """
    opens pickled data object
    
    name : string
    	named of saved data object
    	
    returns : list of tuple of datetime and np array 

    """
    with open(name, "rb") as fp:   # Unpickling
        data = pickle.load(fp)
    return data

class simplexnoiseLoader(Dataset):
    def __init__(self, path, mode):
        """
        dataset class for train loop
        path: str
            path to image and target folder
        mode: str
            train, val, test
        """
        self.mode = mode
        self.path = path

        # take 80% of data as train and val, take 20% of data for testing
        # get list of all image paths in directory
        images = os.listdir(os.path.join(self.path, "images"))
        paths = [os.path.join(os.path.join(self.path, "images"), item) for item in images]
        # take 80% of data as training
        criterion = round(len(paths) * 0.9) # turn off for aletsch as is already splitted, change for 0.8
        #change to one because i separate the test data to 2018 and 2019
        paths = paths[0:criterion]
        self.codigo = images
        self.images = paths

        targets = os.listdir(os.path.join(self.path, "targets"))
        paths = [os.path.join(os.path.join(self.path, "targets"), item) for item in targets]
        paths = paths[0:criterion]
        self.targets = paths

        r = 100
        #randomly select r indexes from the list paths and e
        indexes = np.random.choice(len(paths), r, replace = False)
        #for train and val
        self.images[0:criterion]
        if self.mode == "train":
            # take 90% of data as training
            criterion = round(len(self.images) * 0.95)
            self.images = self.images[0:criterion]
            self.targets = self.targets[0:criterion]

        if self.mode == "val":
            # take 10% as validation set
            criterion = round(len(paths) * 0.95) # cero because i want to test all
            self.images = self.images[criterion:]
            self.targets = self.targets[criterion:]

        if self.mode == "test":
            images = os.listdir(os.path.join(self.path, "images"))
            paths = [os.path.join(os.path.join(self.path, "images"), item) for item in images]
            targets = os.listdir(os.path.join(self.path, "targets"))
            paths = [os.path.join(os.path.join(self.path, "targets"), item) for item in targets]
            # get list of all image paths in directory
            criterion = round(len(paths) * 0.9) # cero because i want to test all
            self.images = images[criterion:]
            self.targets = targets[criterion:]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        returns datum for training

        idx: int
            index to datum
        returns: torch.tensor
                image and targets
        """
        if is_tensor(idx):
            idx = idx.tolist()

        try:
            # get data in tensor format
            inpt = openData(self.images[idx])
            #inpt = inpt[:, 2, :, :]
            target = openData(self.targets[idx])
            
        except:
            # get data in tensor format
            index = np.random.randint(self.__len__())
            inpt = openData(self.images[index])
            #inpt = inpt[:, 2, :, :]
            target = openData(self.targets[index])

        return inpt, target
