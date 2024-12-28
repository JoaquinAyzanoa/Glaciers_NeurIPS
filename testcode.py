import pickle

file = r'C:\FAU\3rd SEM\Glaciers_NeurIPS\datasets\parvati\Test\TemperatureDataPatched\images\0'
image = r'C:\FAU\3rd SEM\Glaciers_NeurIPS\datasets\parvati\alignedAveragedDataNDSI\9'
#load pickle file
"""with open(file, 'rb') as fp:
    data = pickle.load(fp)
    print(type(data))
    print(data)"""

with open(image, 'rb') as fp:
    data = pickle.load(fp)
    print(type(data))
    print(data.shape)
    print(data)