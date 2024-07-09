
# --------------- INTRO ---------------

# This version is working based on v0. The modifications involve:
# - using joblib to process parallel computation.

# ------------------------------

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, svm
from joblib import Parallel, delayed
sys.path.append('D:\Sina Tabeiy\Project\Functions')
import featurextractor


# --------------- Load Data ---------------

# ----- Define variables -----
file_directory = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Pre Data'
measurements = ['pctToeOff', 'pctSimpleAppuie', 'distPas', 'distFoulee', 'tempsFoulee']
output_dir = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\CSV Output'
significance_value = -1
# ----------

# The output data will be in the order of all pre intervention files and then all post intervention files. 
# Since the folder only contain Pre data so we are gonna have the pre data only. 
all_data = featurextractor.feature_extractor(file_directory, measurements, output_dir, separate_legs = False)
#all_data = np.asarray(all_data)

# ----- Load the final result and label the data -----
gps = pd.read_csv(r'D:\Sina Tabeiy\Project\Results\GPS_results\GPS_output.csv')
difference = np.diff(gps, axis=1)
labels = np.where(difference < significance_value, 1, 0).flatten()
# ACTION REQUIRED: IF RUNNING WITH separate_legs = False, DEACTIVATE THE FOLLOWING LINE. OTHERWISE, KEEP IT ACTIVATED.
#labels = np.repeat(labels, 2)

# --------------- ML algorithm: Support Vector Machine (SVM) ---------------
start_time = time.time()
# ----- Train/Test Data split -----
x_train, x_test, y_train, y_test = train_test_split(all_data, labels, test_size = 0.3, random_state = 0)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# ----- Evaluate all types -----
print('************************************')
kernels_types = ['linear', 'poly', 'rbf', 'sigmoid']

def train_and_evaluate(kernel_type):
    SVM = svm.SVC(kernel=kernel_type)
    SVM.fit(x_train, y_train)
    y_hat = SVM.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_hat)
    print("The kernel is: %s" %kernel_type)
    print(metrics.classification_report(y_test, y_hat))
    return kernel_type.upper(), acc


results = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(kernel) for kernel in kernels_types)

for kernel_type, acc in results:
    print("The accuracy of the {} model is: {:.3f} %".format(kernel_type, acc*100))
    print('************************************')

end_time = time.time()
print("The run time of the code is %f seconds" %(end_time - start_time))