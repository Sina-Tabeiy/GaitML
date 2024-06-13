import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt


# RREAD .mat    FILES IN PYTHON

def access_struct (data,structs):
    for struct in structs:
        
        if isinstance(data,np.ndarray) and data.dtype.names is not None:
            data = data[0,0][struct]
        else:
            data = data[struct]
        
    return data




"""

def access_struct (data,structs):
    for struct in structs:
        
        if isinstance(data,np.ndarray):
            if data.dtype.names is not None:
                data = data[struct]
            else:
                if data.ndim == 0:
                    data = data.item()
                else:
                    data = data[struct]

        elif isinstance(data,dict):
            data = data[struct]
                
    return data

"""
joint_data = np.empty((100,0))

# MAKE SURE YOU DO NOT SQUEEZE DATA BY  squeeze_me= True. OTHERWISE THE CODE RUNS INTO ERRORS
file_path = r"D:\Sina Tabeiy\Clustering Project\Lokomat Data (matfiles)\patient1_PostLokomat.mat"
data = loadmat(file_path)

side_structs = ['Right', 'Left']
measurements = ['angAtFullCycle', 'pctToeOff', 'pctToeOffOppose']

# WRITE THE NAME OF STRUCTS YOU WANT TO INCLUDE, DO NOT FORGER TO PUT THEM IN ORDERD
for side_struct in side_structs:    
    for measurement in measurements:
    
        structs = ['c', 'results', side_struct, measurement]
        all_data = access_struct(data,structs)
        
        if measurement == 'angAtFullCycle':

            joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']
            #kin_items = ['Hip']
            sides = side_struct[0]

            #joint_data = np.empty((100,0))
            #list_joint_kin = []

            for joint in (joint_names):
                for side in sides:
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0]
                    #list_joint_kin = joint_kin.flatten(order = "F")
                    joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                    #joint_kin1 = [item for sublist in list_joint_kin for item in sublist]
                    #df.append(joint_with_side)
                    
                    #joint_data.append(joint_kin)
                    joint_data = np.concatenate((joint_data,joint_kin), axis = 1)
        
        else:

            variable = all_data[0][0]
            filler = np.full((99,1), np.nan)
            variable = np.vstack((variable, filler))
            joint_data = np.concatenate((joint_data,variable), axis = 1)
            
            

#print(all_data[0][0])

pd.DataFrame(joint_data).to_csv('output1.csv')
print("The data successfully saved!")
    
