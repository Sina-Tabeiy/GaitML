import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

dataset = np.random.randint(5,50, size = 25)
dataset = dataset.reshape(-1,1)
# print(dataset)

scaler = StandardScaler()
dataset_fit = scaler.fit(dataset)
dateset_fit = dataset_fit.mean_()
dataset_fittransform = scaler.fit_transform(dataset)


print("***********")
print("fit:" )
print(dataset_fit)
# print("***********")
# print("fit_transform")
# print(dataset_fittransform)