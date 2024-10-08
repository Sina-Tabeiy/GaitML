import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from scipy import interpolate
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

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

directory = r"D:\Sina Tabeiy\Clustering Project\Lokomat Data (matfiles)\New sample"


pre_files = [f for f in os.listdir(directory) if f.endswith("eLokomat.mat")]
post_files = [f for f in os.listdir(directory) if f.endswith("stLokomat.mat")]

# --------------- This part prioritize the order of the files ---------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
mat_files_pre_sorted = sorted(pre_files, key=natural_sort_key)
mat_files_post_sorted = sorted(post_files, key=natural_sort_key)
mat_files_sorted = mat_files_pre_sorted + mat_files_post_sorted
# ----------------------------------------------------------------------------

for file_num, file in enumerate(mat_files_sorted):
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
    
    joint_data.to_csv(r'.\Results\GPS_results\Subject%d_Lokomat.csv' %(file_num+1), index = False)

    # if file_num < ((len(list(enumerate(mat_files)))+1)//2):
    #     joint_data.to_csv(r'.\Results\GPS_results\Subject%d_PreLokomat.csv' %(file_num+1), index = False)

    # else:
    #     joint_data.to_csv(r'.\Results\GPS_results\Subject%d_PostLokomat.csv' %(file_num + 1 - (((len(list(enumerate(mat_files)))+1)//2))), index = False)
    
    print("The data is successfully saved!")


# -------------------- This part calculates the GPS -----------------------------

def reload_data(file_path):

    csv = pd.read_csv(file_path)
    dependent_variables = csv
    independent_variable = pd.Series(range(len(dependent_variables)))     
    interpolation_function = pd.DataFrame()
    interpolated_dependent_variable = pd.DataFrame()

    for i in range(len(dependent_variables.columns)):
        scaled_independent_variable = np.linspace(0, len(dependent_variables) - 1, num=51)
        interpolation_function = interpolate.interp1d(independent_variable, dependent_variables.iloc[:, i], kind='cubic')
        interpolated_dependent_variable[f'{i + 1}'] = interpolation_function(scaled_independent_variable)
    
    return interpolated_dependent_variable

directory_str = r"D:\Sina Tabeiy\Clustering Project\Results\GPS_results"

# an example for reference 
# refrence = [27.46143791, -0.311111111, 8.68496732, 25.92026144, -1.811764706, 4.474509804, 21.30522876, 1.612745098, -4.044771242, 27.46143791, -0.311111111, 8.68496732, 25.92026144, -1.811764706, 4.474509804]
reference = pd.read_csv(r'D:\Sina Tabeiy\Clustering Project\TD Data (matfiles)\average.csv')
reference = list(reference.iloc[:,1])
all_GPS = []

# --------------- This part prioritize the order of the files ---------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
file_list = os.listdir(directory_str)
file_list = [f for f in file_list if f.endswith('mat.csv')]
file_list_sorted = sorted(file_list, key=natural_sort_key)
# ----------------------------------------------------------------------------

print("--------------------------------")
print("Data Loaded! Waiting for analysis")
for index, file in enumerate(file_list_sorted, start = 0):
    all_GVS = []
    file_path = os.path.join(directory_str, file)
    kinematics = reload_data(file_path)

    for i in range(15):
        diffrences = kinematics.iloc[:,i] - reference[i]
        GVS = np.sqrt(np.mean(diffrences**2))
        all_GVS.append(GVS)
    
    GPS = np.mean(all_GVS)
    all_GPS.append(GPS)


# print(all_GPS)
# reshaped_array = np.reshape(all_GPS, (22, -1))
# print(reshaped_array)

reshaped_array =[]
for i in range(len(all_GPS)//2):
    output = [all_GPS[i], all_GPS[i+(len(all_GPS)//2)]]
    reshaped_array = reshaped_array + output

reshaped_array = np.reshape(reshaped_array,(-1,2))

GPS_output = pd.DataFrame(reshaped_array)
GPS_output.columns = ['Pre', 'Post']
GPS_output.to_csv(r'.\Results\GPS_results\GPS_output.csv', index = False)
print("--------------------------------")
print("Analysis Done!")


# ----------------       Apply SK-learn kmeans to the data      ----------------
def apply_sk_kmeans(data, max_k):
    
    inertia = []
    silhouette_scores = []
    result = []
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    for k in range(2, max_k):
        model = KMeans(n_clusters=k, init='k-means++', n_init=30, random_state=0)
        model.fit(normalized_data)
        cluster_labels = model.fit_predict(normalized_data)
        
        #labels = model.labels_
        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_score(normalized_data, cluster_labels))
        
        if k ==2:
            for i in range(data.shape[0]//2):
                output = [f'Subject {i+1}', cluster_labels[i], cluster_labels[i+(data.shape[0]//2)]]
                result = result + output

            result = pd.DataFrame(np.reshape(result,(-1,3)))
            result.columns = ['Sub No.', 'Pre', 'Post']
            result.to_csv(r'D:\Sina Tabeiy\Clustering Project\Results\GPS_results\GPS_Clustering.csv', index = False)

    # plt.plot(range(2, max_k), inertia, marker='o', linestyle='--')
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.show()

    # # Plotting the Silhouette Scores
    # #plt.figure(figsize=(10, 5))
    # plt.plot(range(2, max_k), silhouette_scores, marker='o', linestyle='--')
    # plt.title('Silhouette Scores for Different k')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.show()

    
# file = r'D:\Sina Tabeiy\Clustering Project\Results\GPS_results\GPS_output.csv'
# data = pd.read_csv(file)
data = np.array(all_GPS).reshape(-1,1)
apply_sk_kmeans(data, 5)
