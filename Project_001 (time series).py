import pandas as pd
import numpy as np
import os
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





# MAKE SURE YOU DO NOT SQUEEZE DATA BY  squeeze_me= True. OTHERWISE THE CODE RUNS INTO ERRORS
#file_path = r"D:\Sina Tabeiy\Clustering Project\Lokomat Data (matfiles)\patient1_PostLokomat.mat"
#data = loadmat(file_path)
directory = r"D:\Sina Tabeiy\Clustering Project\Lokomat Data (matfiles)"

mat_files = [f for f in os.listdir(directory) if (f.endswith("ELokomat.mat") or f.endswith("eLokomat.mat"))]

for index, file in enumerate(mat_files):
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

                joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']
                #kin_items = ['Hip']
                sides = side_struct[0]

                for joint in (joint_names):
                    for side in sides:

                        joint_with_side = side + joint
                        joint_kin = all_data[0,0][joint_with_side][0][0]
                        #list_joint_kin = joint_kin.flatten(order = "F")
                        joint_kin = np.reshape(joint_kin, (100,3), order = 'F')
                        #joint_kin1 = [item for sublist in list_joint_kin for item in sublist]
                        #df.append(joint_with_side)
                        
                        #joint_data.append(joint_kin)
                        #joint_kin = [joint_with_side, joint_with_side, joint_with_side].append(joint_kin)
                        
                        joint_data = np.concatenate((joint_data, joint_kin), axis = 1)
         
        """   

            else:

                variable = all_data[0][0]
                filler = np.full((99,1), np.nan)
                variable = np.vstack((variable, filler))
                joint_data = np.concatenate((joint_data,variable), axis = 1)
        """  
           

    #print(all_data[0][0])

    pd.DataFrame(joint_data).to_csv('Subject%d_PreLokomat.csv' %index, index = False)
    print("The data is successfully saved!")



#------------------------------------    This Part loads the previously saved data, and runs the algorithm     -----------------------------------


# ----------------       Reloading data      ----------------
def load_data(directory_str):

    combined_df = pd.DataFrame()
    csv_files = [f for f in os.listdir(directory_str) if f.endswith("t.csv")]

    for index, file in enumerate(csv_files):
        
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
        combined_df.to_csv("allfiles.csv")
    
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
    
    data = data.values
    data = data.reshape(data.shape[0]//100, 100, data.shape[1])

    for k in range(2,max_k):
        model = TimeSeriesKMeans(n_clusters=k, metric = "dtw")
        labels = model.fit_predict(data)
        #silhouette_scores.append(silhouette_score(data, labels, metric = 'dtw'))

        plt.figure()

        for i in range(k):
            plt.subplot(k, 1, i + 1)
            for j in data[labels == i]:
                plt.plot(j[:, 0], "k-", alpha=0.2)  # Plotting only the first feature for simplicity
            plt.plot(model.cluster_centers_[i][:, 0], "r-")
            plt.title(f'Cluster {i + 1}')

            plt.tight_layout()
            plt.show()



directory_str = r'D:\Sina Tabeiy\Clustering Project'
data = load_data(directory_str)
apply_ts_kmeans(data, 5)
