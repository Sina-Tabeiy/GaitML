# --------------- INTRO ---------------

# This version is the main version to modify and find the best solution for the type of study that we have.
# Other versions are just modifications.
# Find the informaition regarding each model in the INTRO of each one.

# ------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics, svm
from scipy.stats import skew, kurtosis
import os
import sys
import time
sys.path.append('D:\Sina Tabeiy\Project\Functions')
import featurextractor


# --------------- Load Data ---------------

# ----- Define variables -----
file_directory = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Pre Data'
measurements = ['pctToeOff', 'pctSimpleAppuie', 'distPas', 'distFoulee', 'tempsFoulee']
#joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']
output_dir = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\CSV Output'
significance_value = -1
# The output data will be in the order of all pre intervention files and then all post intervention files. 
# Since the folder only contain Pre data so we are gonna have the pre data only. 
all_data = featurextractor.feature_extractor(file_directory, measurements, output_dir, separate_legs = False)
df = pd.read_csv(r"D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\CSV Output\all_files.csv")
print(df.describe())
print("Skewness:\n", skew(all_data,axis=0))
print("Kutosis:\n", kurtosis(all_data, axis=0))

# ----- Load the final result and label the data -----
gps = pd.read_csv(r'D:\Sina Tabeiy\Project\Results\GPS_results_all\GPS_output.csv')
diffrence = np.diff(gps,axis=1)
labels = np.where(diffrence < significance_value, 1, 0).flatten()
# ACTION REQUIRED: IF RUNNING WITH separate_legs = False, DEACTIVATE THE FOLLOWING LINE. OTHERWISE, KEEP IT ACTIVATED.
# labels = np.repeat(labels, 2)

"""
labeled_data = np.concatenate((all_data, labels), axis = 1)
pd.DataFrame(labeled_data).to_csv(output_dir + r'\labeled_data.csv', header = False, index = False)
"""

# --------------- ML algorithm: Support Vector Machine (SVM) ---------------
start_time = time.time()
# ----- Train/Test Data split -----
x_train, x_test, y_train, y_test = train_test_split(all_data, labels, test_size = 0.3, random_state = 0)

# ----- Scale the x data -----
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print('Skewness:',skew(all_data))
print('Kurtosis:', kurtosis(all_data))


# ----- Evaluate all types -----
print('************************************')
kernels_types = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel_type in kernels_types:
    SVM = svm.SVC(kernel = kernel_type)
    SVM.fit(x_train, y_train)
    y_hat = SVM.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_hat)
    print("The accuracy of the %s model is: %.3f %%" %(kernel_type.upper(), acc*100))
    print('************************************')
    #print(metrics.classification_report(y_test, y_hat))
end_time = time.time()
print("The run time of the code is %f seconds" %(end_time - start_time))