import torch
from torch.utils import data
import numpy as np
import pickle
import matplotlib.pyplot as plt
from random import randint

# class UrbanSound8KDataset(data.Dataset):
#     def __init__(self, dataset_path, mode):
#         self.dataset = pickle.load(open(dataset_path, 'rb'))
#         self.mode = mode
#
#     def __getitem__(self, index):
#
#         dataset = np.array(self.dataset)
#         label = self.dataset[index]['classID']
#         fname = self.dataset[index]['filename']
#
#         LM = dataset[index]["features"]["logmelspec"]
#         MFCC = dataset[index]["features"]["mfcc"]
#         C = dataset[index]["features"]["chroma"]
#         SC = dataset[index]["features"]["spectral_contrast"]
#         T = dataset[index]["features"]["tonnetz"]
#
#         if self.mode == 'LMC':
#             # Edit here to load and concatenate the neccessary features to
#             # create the LMC feature
#             LMC = np.concatenate((LM, C, SC, T), axis=0)
#             plt.imshow(LM)
#             plt.show()
#             plt.imshow(MFCC)
#             plt.show()
#             plt.imshow(C)
#             plt.show()
#             plt.imshow(SC)
#             plt.show()
#             plt.imshow(T)
#             plt.show()
#             feature = torch.from_numpy(LMC.astype(np.float32)).unsqueeze(0)
#             print(fname)
#             print(label)
#         elif self.mode == 'MC':
#             # Edit here to load and concatenate the neccessary features to
#             # create the MC feature
#             MC = np.concatenate((MFCC, C, SC, T), axis=0)
#             feature = torch.from_numpy(MC.astype(np.float32)).unsqueeze(0)
#         elif self.mode == 'MLMC':
#             # Edit here to load and concatenate the neccessary features to
#             # create the MLMC feature
#             MLMC = np.concatenate((MFCC, LM, C, SC, T), axis=0)
#             feature = torch.from_numpy(MLMC.astype(np.float32)).unsqueeze(0)
#
#         return feature, label, fname
#
#     def __len__(self):
#         return len(self.dataset)

class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):

        # Load the dataset
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):

        # Extract the necessary features from the loaded dataset
        LM = self.dataset[index]["features"]["logmelspec"]
        MFCC = self.dataset[index]["features"]["mfcc"]
        C = self.dataset[index]["features"]["chroma"]
        SC = self.dataset[index]["features"]["spectral_contrast"]
        T = self.dataset[index]["features"]["tonnetz"]

        # Appropriately prepare the data given the selected mode, based on the specifications of the paper
        if self.mode == 'LMC':
            LMC = np.concatenate((LM, C, SC, T), axis=0)
            print (LMC)
            print()
            LMC = LMC*randint(1,11)
            feature = torch.from_numpy(LMC.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            MC = np.concatenate((MFCC, C, SC, T), axis=0)
            feature = torch.from_numpy(MC.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            MLMC = np.concatenate((MFCC, LM, C, SC, T), axis=0)
            feature = torch.from_numpy(MLMC.astype(np.float32)).unsqueeze(0)
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']

        return feature, label, fname, index

dataset_train_LMC = UrbanSound8KDataset("./UrbanSound8K_test.pkl", "LMC")
dataset_train_MC = UrbanSound8KDataset("./UrbanSound8K_test.pkl", "MC")



# for i in range (0,20):
#     a = randint(0,5395)
#     print (a)
#     plt.imshow(dataset_train_LMC.__getitem__(a)[0].squeeze())
#     plt.hlines(59, 0, 40)
#     plt.hlines(66, 0, 40)
#     plt.hlines(71, 0, 40)
#     plt.show()

# set = np.array([3064, 3065, (3066), (3067), (3068), (3069), (3070), 3071, 3072])
#
# set = np.array([(2244), (2245), (2246), (2247), (2248), (2249), (2250)])
# for i in set:
#     dataset_train_LMC.__getitem__(i)[0]

i = 4690
dataset_train_LMC.__getitem__(i)[0]


# labels = np.zeros((10,1))
# print(labels)
# for i in range(5395):
#     labels[dataset_train_LMC.__getitem__(i)[1]] += 1
#
# print(labels)
#
# LMC = pickle.load(open("./TSCNN_store_LMC.pkl", 'rb'))
# MC = pickle.load(open("./TSCNN_store_MC.pkl", 'rb'))
# LMC = LMC.cpu()
# MC = MC.cpu()
