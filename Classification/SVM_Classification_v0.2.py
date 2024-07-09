# --------------- INTRO ---------------

# This version is working based on v0.0, The modifications involve:
# - Adding gridsearch to tune hyperparameters.
# - Adding StandardScaler and MinMaxScaler (Normalizer).
# - Adding Cross validation in GridSearchCV.
# ------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics, svm
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

# ----------
# The output data will be in the order of all pre intervention files and then all post intervention files. 
# Since the folder only contain Pre data so we are gonna have the pre data only. 
all_data = featurextractor.feature_extractor(file_directory, measurements, output_dir, separate_legs = False)


# ----- Load the final result and label the data -----
gps = pd.read_csv(r'D:\Sina Tabeiy\Project\Results\GPS_results\GPS_output.csv')
diffrence = np.diff(gps,axis=1)
labels = np.where(diffrence < significance_value, 1, 0).flatten()
# ACTION REQUIRED: IF RUNNING WITH separate_legs = False, DEACTIVATE THE FOLLOWING LINE. OTHERWISE, KEEP IT ACTIVATED.
#labels = np.repeat(labels, 2)

if isinstance(all_data, pd.DataFrame):
    all_data = all_data.values
# --------------- ML algorithm: Support Vector Machine (SVM) ---------------

start_time = time.time()

# ----- Train / test Split -----
x_train, x_test, y_train, y_test = train_test_split(all_data, labels, test_size = 0.2, random_state = 0)

# ----- Scale the data -----
normalizer = MinMaxScaler()
scaler = StandardScaler()
# x_train = normalizer.fit_transform(x_train)
# x_test = normalizer.transform(x_test)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ----- Grid Search -----
print('************************************')
print('Grid Search initiated.')

SVM = svm.SVC()

parameter_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

grid_search = GridSearchCV(SVM, parameter_grid, n_jobs = -1, refit = True, verbose = 3, cv = 5)
grid_search.fit(x_train, y_train)

print('********** Best parameters ********** ')
print("The best parameters are: ", grid_search.best_params_)
print('********** Best estimator ********** ')
print("The best estimator is:", grid_search.best_estimator_)
print("Test set score: ", grid_search.score(x_test, y_test))
#print(grid_search.best_score_)
#accuracies.append(metrics.classification_report(y_test, grid_search.best_estimator_.predict(x_test)))

end_time = time.time()
print("The run time of the code is %f seconds" %(end_time - start_time))
