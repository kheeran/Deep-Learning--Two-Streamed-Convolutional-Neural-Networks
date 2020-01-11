import torch
from torch.utils import data
import numpy as np
import pickle
import matplotlib.pyplot as plt
from random import randint

results_epoch_LMC = pickle.load(open("TSCNN_store_LMC_Final.pkl", "rb"))
results_epoch_MC = pickle.load(open("TSCNN_store_MC_Final.pkl", "rb"))
# results_epoch_TSCNN = pickle.load(open("TSCNN_store_TSCNN.pkl", "rb"))

Final_LMC = results_epoch_LMC[49]
Final_MC = results_epoch_MC[49]

print(len(Final_LMC['preds']))

# indices where LMC and MC both classify correctly
LMC_vs_MC_correct = np.logical_and((np.array(Final_LMC['preds']) == np.array(Final_LMC["labels"])),(np.array(Final_MC['preds'] == np.array(Final_LMC["labels"]))))
x = np.where(LMC_vs_MC_correct)[0]
print(x.shape)

# indices where LMC and MC both classify wrongly
LMC_vs_MC_wrong = np.logical_and((np.array(Final_LMC['preds']) != np.array(Final_LMC["labels"])),(np.array(Final_MC['preds'] != np.array(Final_LMC["labels"]))))
print((np.where(LMC_vs_MC_wrong))[0].shape)

# indices where either LMC is correct and MC is wrong
LMC_vs_MC_LMC = np.logical_and((np.array(Final_LMC['preds']) == np.array(Final_LMC["labels"])),(np.array(Final_MC['preds'] != np.array(Final_LMC["labels"]))))
print((np.where(LMC_vs_MC_LMC))[0].shape)

# indices where either LMC is wrong and MC is correct
LMC_vs_MC_MC = np.logical_and((np.array(Final_LMC['preds']) != np.array(Final_LMC["labels"])),(np.array(Final_MC['preds'] == np.array(Final_LMC["labels"]))))
print((np.where(LMC_vs_MC_MC))[0].shape)
