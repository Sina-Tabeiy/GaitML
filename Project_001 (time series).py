import pandas as pd
import numpy as np
import os
import re
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
#data = loadmat(file_path)

directory = r"D:\Sina Tabeiy\Clustering Project\Lokomat Data (matfiles)"

# Ensures that first the "pre" is analyzed and then the "post training" data.
pre_files = [f for f in os.listdir(directory) if f.endswith("eLokomat.mat")]
post_files = [f for f in os.listdir(directory) if f.endswith("stLokomat.mat")]

# --------------- This part prioritize the order of the files ---------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
mat_files_pre_sorted = sorted(pre_files, key=natural_sort_key)
mat_files_post_sorted = sorted(post_files, key=natural_sort_key)
mat_files_sorted = mat_files_pre_sorted + mat_files_post_sorted
# ----------------------------------------------------------------------------


for file_number, file in enumerate(mat_files_sorted, start = 0):
    file_path = str()
    file_path = os.path.join(directory, file) 
    data = load_data(file_path)

    side_structs = ['Right', 'Left']
    measurements = ['angAtFullCycle', 'pctToeOff', 'pctToeOffOppose']
    joint_data = np.empty((100,0))

    # WRITE THE NAME OF STRUCTS YOU WANT TO INCLUDE, DO NOT FORGER TO PUT THEM IN ORDERD
    for side_struct in side_structs:    
        for measurement in measurements:
        
            structs = ['c', 'results', side_struct, measurement]
            all_data = access_struct(data,structs)
            
            if measurement == 'angAtFullCycle':

                #joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']
                joint_names = ['Hip','Knee', 'Ankle']
                sides = side_struct[0]

                for joint in (joint_names):
                    for side in sides:
                        
                        joint_with_side = side + joint
                        # \\\\\\\\\\\ This one was changed to have the analysis based on the result of flx/ext ///////////
                        # \\\\\\ joint_kin = all_data[0,0][joint_with_side][0][0] //////
                        # \\\\\\ joint_kin = np.reshape(joint_kin, (100,1), order = 'F') //////
                        joint_kin = all_data[0,0][joint_with_side][0][0][:,0]
                        joint_kin = np.reshape(joint_kin, (100,1), order = 'F')

                        joint_data = np.concatenate((joint_data, joint_kin), axis = 1)


            # ------------ This line is only for calculated parameters e.g. cadence ------------
            # else: 

            #     variable = all_data[0][0]
            #     filler = np.full((99,1), np.nan)
            #     variable = np.vstack((variable, filler))
            #     joint_data = np.concatenate((joint_data,variable), axis = 1)
            # ------------------------------------------------------------------------------------

    
    pd.DataFrame(joint_data).to_csv(r'.\Results\Time ceries clustering_results\Subject%d_Lokomat.csv' % (file_number +1), index = False)
    print("The data is successfully saved!")



# ---------------------- This Part loads the previously saved data, and runs the algorithm ----------------------


# ----------------       Reloading data      ----------------
def reload_data(directory_str):

    combined_df = pd.DataFrame()
    #csv_files = [f for f in os.listdir(directory_str) if f.endswith("mat.csv")]
    file_list = os.listdir(directory_str)
    file_list = [f for f in file_list if f.endswith('mat.csv')]
    csv_files = sorted(file_list, key=natural_sort_key)

    for file_number, file in enumerate(csv_files):
        
        file_path = os.path.join(directory_str, file)
        csv = pd.read_csv(file_path)
        dependent_variables = csv
        independent_variable = pd.Series(range(len(dependent_variables)))     
        interpolation_function = pd.DataFrame()
        interpolated_dependent_variable = pd.DataFrame()

        for i in range(len(dependent_variables.columns)):
            scaled_independent_variable = np.linspace(0, len(dependent_variables) - 1, num=100)
            interpolation_function = interpolate.interp1d(independent_variable, dependent_variables.iloc[:, i], kind='cubic')
            interpolated_dependent_variable[f'{i + 1}'] = interpolation_function(scaled_independent_variable)
        
        # /?/ Decide on the "feature_range" whether the range is suitable for gait analysis or not
        #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler = preprocessing.StandardScaler()
        scaled_dependent_variables = pd.DataFrame(scaler.fit_transform(interpolated_dependent_variable))
        #scaled_dependent_variables.insert(0, 'Subject ID', index + 1)
        scaled_dependent_variables.columns = scaled_dependent_variables.columns.astype(str)

        combined_df = pd.concat([combined_df, scaled_dependent_variables], ignore_index=True)
        #combined_df.to_csv("allfiles.csv")
    
    print("--------------------------------")
    print("The shape of data is: {}".format(combined_df.shape))
    print("--------------------------------")
    print("The data is now ready to be analyzed!")
    return combined_df

# ----------------       This applies SK-learn kmeans to the data      ----------------
def apply_sk_kmeans(data, max_k):
    
    from sklearn.metrics import silhouette_score
    inertia = []
    silhouette_scores = []

    for k in range(2, max_k):

        model = KMeans(n_clusters=k, init='k-means++', n_init=30, random_state=0)
        model.fit(data)
        labels = model.labels_
        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
        
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        # Create a scatter plot of the clustered data
        plt.figure(figsize=(10, 7))
        for cluster in range(k):
            plt.scatter(reduced_data[labels == cluster, 0], reduced_data[labels == cluster, 1], label=f'Cluster {cluster}')
        plt.title('K-means Clustering Results (PCA-reduced data)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()


"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_k), inertia, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Plotting the Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_k), silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Scores for Different k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
"""


# ----------------       This applies TS-learn kmeans to the data      ----------------

def apply_ts_kmeans (data, max_k):

    from tslearn.clustering import TimeSeriesKMeans, silhouette_score
    result = []

    data = data.values
    data = data.reshape(data.shape[0]//100, 100, data.shape[1])

    for k in range(2,max_k):
        model = TimeSeriesKMeans(n_clusters=k, metric = "dtw", random_state = 0)
        labels = model.fit_predict(data)
        #silhouette_scores.append(silhouette_score(data, labels, metric = 'dtw'))
        if k ==2:
            for i in range(data.shape[0]//2):
                output = [f'Subject {i+1}', labels[i], labels[i+(data.shape[0]//2)]]
                result = result + output
                

        plt.figure()
        for i in range(k):
            plt.subplot(k, 1, i + 1)
            for j in data[labels == i]:
                plt.plot(j[:, 0], "k-", alpha=0.2)
            plt.plot(model.cluster_centers_[i][:, 0], "r-")
            plt.title(f'Cluster {i + 1}')
        
    plt.tight_layout()
    plt.show()
    pd.DataFrame(np.reshape(result,(-1,3))).to_csv(r'D:\Sina Tabeiy\Clustering Project\Results\Time ceries clustering_results\outcome.csv')


directory_str = r'D:\Sina Tabeiy\Clustering Project\Results\Time ceries clustering_results'
data = reload_data(directory_str)
apply_ts_kmeans(data, 5)
