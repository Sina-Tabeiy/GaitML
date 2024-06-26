import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from scipy import interpolate

#------------------------------------    This Part loads the data, extract the features and save them in an excel file     -----------------------------------
# RREAD .mat    FILES IN PYTHON
# Identifying the dataset
def load_data(file_path):

    data = loadmat(file_path)

    print("--------------------------------")
    print("The data is now loaded!")

    return data


def access_struct (data,structs):
    for struct in structs:
        
        if isinstance(data,np.ndarray) and data.dtype.names is not None:
            data = data[0,0][struct]
        else:
            data = data[struct]
        
    return data

# MAKE SURE YOU DO NOT SQUEEZE DATA BY  squeeze_me= True. OTHERWISE THE CODE RUNS INTO ERRORS
#file_path = r"D:\Sina Tabeiy\Clustering Project\Lokomat Data (matfiles)\patient1_PostLokomat.mat"

directory = r"D:\Sina Tabeiy\Clustering Project\TD Data (matfiles)"


CTL_files = [f for f in os.listdir(directory) if f.endswith(".mat")]

# --------------- This part prioritize the order of the files ---------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
CTL_files_sorted = sorted(CTL_files, key=natural_sort_key)
# ---------------------------------------------------------------------------

for file_num, file in enumerate(CTL_files_sorted):
    file_path = str()
    file_path = os.path.join(directory, file) 
    data = load_data(file_path)

    side_structs = ['Right', 'Left']
    joint_data = np.empty((100,0))

    # WRITE THE NAME OF STRUCTS YOU WANT TO INCLUDE, DO NOT FORGER TO PUT THEM IN ORDER
    for side_struct in side_structs:    

    
        structs = ['c', 'results', side_struct, 'angAtFullCycle']
        all_data = access_struct(data,structs)

        joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress']
        sides = side_struct[0]

        if side_struct == 'Left':
                joint_with_side = 'LPelvis'
                joint_kin = all_data[0,0][joint_with_side][0][0]
                joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

        for joint in (joint_names):
            if joint == 'Knee':
                for side in sides:
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,0]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)
                
            elif joint == 'Ankle':
                for side in sides:
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,0]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

            elif joint == 'FootProgress':
                for side in sides:
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0][:,2]
                    joint_kin = np.reshape(joint_kin, (100,1), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)
            
            else: # Consider hip only
                for side in sides:
                    joint_with_side = side + joint
                    joint_kin = all_data[0,0][joint_with_side][0][0]
                    joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                    joint_data = np.concatenate((joint_data, joint_kin), axis = 1)

    joint_data = pd.DataFrame(joint_data)
    joint_data.columns = ['R_Hip flex/ext', 'R_Hip abd/add', 'R_Hip int/ext rotation', 'R_Knee flx/ext',
                            'R_Ankle dorsi/plantar flx', 'R_foot progression', 'L_Pelvis', 'L_Pelvis', 'L_Pelvis', 'L_Hip flex/ext',
                            'L_Hip abd/add', 'L_Hip int/ext rotation', 'L_Knee flx/ext', 'L_Ankle dorsi/plantar flx', 'L_foot progression']
    
    joint_data.to_csv(r'D:\Sina Tabeiy\Clustering Project\TD Data (matfiles)\CSV\TD_Subject%d.csv' %(file_num+1), index = False)

    # if file_num < ((len(list(enumerate(mat_files)))+1)//2):
    #     joint_data.to_csv(r'.\Results\GPS_results\Subject%d_PreLokomat.csv' %(file_num+1), index = False)

    # else:
    #     joint_data.to_csv(r'.\Results\GPS_results\Subject%d_PostLokomat.csv' %(file_num + 1 - (((len(list(enumerate(mat_files)))+1)//2))), index = False)
    
    print("The data is successfully saved!")

directory = 'D:\Sina Tabeiy\Clustering Project\TD Data (matfiles)\CSV'
dfs = []


for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        dfs.append(df)


combined_df = pd.concat(dfs)
mean_df = combined_df.mean()
mean_df.to_frame().T
mean_df.to_csv(r'D:\Sina Tabeiy\Clustering Project\TD Data (matfiles)\average.csv', header = True)

print('Average columns saved')