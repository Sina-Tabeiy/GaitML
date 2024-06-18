import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
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

mat_files = [f for f in os.listdir(directory) if (f.endswith("ELokomat.mat") or f.endswith("eLokomat.mat"))]

for index, file in enumerate(mat_files):
    file_path = str()
    file_path = os.path.join(directory, file) 
    data = load_data(file_path)

    side_structs = ['Right', 'Left']
    measurements = ['pctToeOff', 'pctToeOffOppose', 'pctContactTalOppose', 'pctSimpleAppuie', 'distPas', 'distFoulee', 'tempsFoulee', 'vitFoulee', 'vitCadencePasParMinute' ]
    joint_data = pd.DataFrame()

    # WRITE THE NAME OF STRUCTS YOU WANT TO INCLUDE, DO NOT FORGER TO PUT THEM IN ORDERD
    for side_struct in side_structs:
        for measurement in measurements:
        
            structs = ['c', 'results', side_struct, measurement]
            all_data = access_struct(data,structs)
            variable = all_data[0]
            #joint_data.append(variable)  # Initialize joint_data with the first variable
            joint_data = pd.concat([joint_data, pd.Series(variable, name=f"{side_struct}_{measurement}")], axis = 1)
            

    pd.DataFrame(joint_data).to_csv('Subject%d_PreLokomat.csv' %index, index = False)
    print("The data is successfully saved!")


#------------------------------------    This Part loads the previously saved data, and runs the algorithm     -----------------------------------


# ----------------       Reloading data      ----------------
def load_data(directory_str):

    combined_df = pd.DataFrame()
    csv_files = [f for f in os.listdir(directory_str) if f.endswith("t.csv")]

    for index, file in enumerate(csv_files):
        
        file_path = os.path.join(directory_str, file)
        dependent_variables = pd.read_csv(file_path)    
        combined_df = pd.concat([combined_df, dependent_variables], ignore_index=True)

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

    scaler = preprocessing.StandardScaler()
    normalized_data = scaler.fit_transform(data)
    #pd.DataFrame(normalized_data).to_csv("normalized_allfiles.csv")

    for k in range(2, max_k):

        model = KMeans(n_clusters=k, init='k-means++', n_init=30, random_state=0)
        model.fit(normalized_data)
        cluster_labels = model.fit_predict(normalized_data)
        data['Cluster'] = cluster_labels
        #labels = model.labels_
        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_score(normalized_data, cluster_labels))
        

        # ------------------------ Visualisation ---------------------------
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(normalized_data)
        plot_data = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
        plot_data['Cluster'] = cluster_labels

        plt.figure()
        for cluster in range(k):
            cluster_data = plot_data[plot_data['Cluster'] == cluster]
            plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster + 1}')
            #plt.scatter(reduced_data[labels == cluster, 0], reduced_data[labels == cluster, 1], label=f'Cluster {cluster}')
        plt.title('K-means Clustering Results')
        plt.legend()
        plt.show()


    plt.plot(range(2, max_k), inertia, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Plotting the Silhouette Scores
    #plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_k), silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Scores for Different k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


directory_str = r'D:\Sina Tabeiy\Clustering Project'
data = load_data(directory_str)
apply_sk_kmeans(data, 5)
