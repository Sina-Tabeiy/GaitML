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
# WRITE THE NAME OF STRUCTS YOU WANT TO INCLUDE, DO NOT FORGER TO PUT THEM IN ORDERD
for side_struct in side_structs:    
    structs = ['c', 'results', side_struct, 'angAtFullCycle']
    all_data = access_struct(data,structs)

    kin_items = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']
    #kin_items = ['Hip']
    sides = side_struct[0]

    #joint_data = np.empty((100,0))
    #list_joint_kin = []

    for joint in (kin_items):
        for side in sides:

            joint_with_side = side + joint
            joint_kin = all_data[0,0][joint_with_side][0][0]
            #list_joint_kin = joint_kin.flatten(order = "F")
            joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
            #joint_kin1 = [item for sublist in list_joint_kin for item in sublist]
            #df.append(joint_with_side)
            
            #joint_data.append(joint_kin)
            joint_data = np.append(joint_data,joint_kin, axis = 1)


#print(joint_data)

pd.DataFrame(joint_data).to_csv('output1.csv')
    
