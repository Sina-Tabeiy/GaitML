import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy import interpolate

# Identifying the dataset
def load_data (directoty_str):

    print("Data recieved!")
    combined_df = []
    csv_files = [f for f in os.listdir(directory_str) if f.endswith(".csv")]

    for index,file in enumerate(csv_files):

        file_path = os.path.join(directory_str, file)
        csv = pd.read_csv(file_path)
        csv.drop(["activity"], axis= 1, inplace= True)
        
        independent_variable = csv.iloc[:,0]
        dependent_variables = csv.iloc[:,1:]

        interpolated_dependent_variable = pd.DataFrame()
        interpolation_funtion = pd.DataFrame()

        for i in range(len(dependent_variables.columns)):
            scaled_independent_variable = np.linspace(0, len(dependent_variables)-1, num = 100)
            interpolation_funtion = interpolate.interp1d(independent_variable, dependent_variables.iloc[:,i], kind= 'cubic')
            interpolated_dependent_variable[f'{i+1}'] = pd.DataFrame(interpolation_funtion(scaled_independent_variable))
        
        # -------------------- Deside on the "feature_range" whether the range is suitable for gait analysis or not-------------
        
        scaler = preprocessing.MinMaxScaler(feature_range= (0,1))
        scaled_dependent_variables = pd.DataFrame(scaler.fit_transform(interpolated_dependent_variable))
        scaled_dependent_variables.insert(0,'Subject ID', index+1)
        #scaled_dependent_variables.columns = scaled_dependent_variables.columns.astype(str)

        combined_df = [scaled_dependent_variables(index)]
    
    print("--------------------------------")
    print("The shape of data is: {}".format(combined_df.shape))
    print("--------------------------------")
    print("The data is now ready to be analyzed!")
    return combined_df

def apply_kmeans(data, max_k):
    
    inertia = []
    silhouette_scores = []

    for k in range(2,max_k):
        model = KMeans(k, init = "k-means++", n_init = 30, random_state = 0)
        prediction = model.fit(data)
        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_scores(data, model.labels_))

    plt.figure(figsize=(10, 5))
    plt.plot(range, inertia, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Plotting the Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(range, silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Scores for Different k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()




directory_str = r'G:\Uni\PhD\K-means\Python code\Example Dataset\Test'
data = load_data(directory_str)
apply_kmeans(data, 3)
#data.to_excel("output.xlsx", index= False)





