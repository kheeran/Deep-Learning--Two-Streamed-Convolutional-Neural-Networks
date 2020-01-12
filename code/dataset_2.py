import torch
from torch.utils import data
import numpy as np
import pickle
import matplotlib.pyplot as plt
from random import randint

results_epoch_LMC = pickle.load(open("TSCNN_store_LMC.pkl", "rb"))

indices = results_epoch_LMC[0]['indices']
labels = results_epoch_LMC[0]['labels']
i = 398
print (indices[i])
print (labels[i])
