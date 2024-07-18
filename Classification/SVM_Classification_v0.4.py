# --------------- INTRO ---------------

# This version is working based on v0.0, The modifications involve:
# - Adding gridsearch to tune hyperparameters.
# - Adding StandardScaler and MinMaxScaler (Normalizer).
# - Adding Cross validation in GridSearchCV.
# - Adding Permutation Importance to the best estimator of the GridSearchCV.
# - Adding coefficient matrix to get the feature importance for the linear model.
# ------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn import metrics, svm

import os
import sys
import time
sys.path.append('D:\Sina Tabeiy\Project\Functions')
import featurextractor


# --------------- Load Data ---------------

# ----- Define variables -----
file_directory = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Pre Data'
# measurements = ['pctToeOff', 'pctSimpleAppuie', 'distPas', 'distFoulee', 'tempsFoulee']
measurements = ['pctToeOff', 'pctSimpleAppuie', 'distPas', 'distFoulee', 'tempsFoulee']
#joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']
side = ['Right', 'Left']
output_dir = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\CSV Output'
significance_value = -1

# ----- Extract variables from .mat files -----
# The output data will be in the order of all pre intervention files and then all post intervention files. 
# Since the folder only contains Pre data so we are gonna have the pre data only. 
all_data = featurextractor.feature_extractor(file_directory, measurements, output_dir, separate_legs = False)
if isinstance(all_data, pd.DataFrame):
    all_data = all_data.values

# ----- Add participants demographic variables -----
demo_var = pd.read_excel(r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Pre Data\Participants_caracteristic_ML.xlsx')
demo_var.drop(['Patient', 'sex','masse', 'taille', 'Diagnostique'], axis = 1, inplace= True)
# demo_var.replace(['Quadri', 'Double Hémi', 'Diplégie'], [3,2,1], inplace=True)
demo_var.replace(['walker', 'cane', 'none'], [3,2,1], inplace=True)
demo_val = demo_var.values
all_data = np.concatenate((all_data, demo_val), axis=1)

# ----- Load the final result and label the data -----
gps = pd.read_csv(r'D:\Sina Tabeiy\Project\Results\GPS_results_all\GPS_output.csv')
diffrence = np.diff(gps,axis=1)
labels = np.where(diffrence < significance_value, 1, 0).flatten()
# ACTION REQUIRED: IF RUNNING WITH separate_legs = False, DEACTIVATE THE FOLLOWING LINE. OTHERWISE, KEEP IT ACTIVATED.
#labels = np.repeat(labels, 2)

# --------------- ML algorithm: Support Vector Machine (SVM) ---------------
start_time = time.time()

# ----- Train / test Split -----
x_train, x_test, y_train, y_test = train_test_split(all_data, labels, test_size = 0.2, random_state = 0)

# ----- Scale the data -----
normalizer = MinMaxScaler()
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ----- Grid Search -----
print('************************************')
print('Grid Search initiated.')

SVM = svm.SVC(kernel = 'rbf',random_state = 0)

parameter_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001],
    # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    # 'degree': [2, 3, 4, 5],  # Only relevant for 'poly' kernel
    # 'coef0': [0.0, 0.1, 0.5, 1.0]  # Only relevant for 'poly' and 'sigmoid' kernels
}

# kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
grid_search = GridSearchCV(SVM, parameter_grid, n_jobs = -1, refit = True, verbose = 3, cv = 10)
grid_search.fit(x_train, y_train)

print('********** Best parameters ********** ')
print("The best parameters are: ", grid_search.best_params_)
print('********** Best estimator ********** ')
print("The best estimator is:", grid_search.best_estimator_)
print("Test set score: ", grid_search.score(x_test, y_test))
#print(grid_search.best_score_)
print(metrics.classification_report(y_test, grid_search.best_estimator_.predict(x_test)))

# ----- Calculate parameters weight fi=or non-linear model -----
fw = permutation_importance(grid_search.best_estimator_, x_test, y_test, n_repeats = 20, n_jobs=-1, random_state = 0)

for i in range(len(fw.importances_mean)):
        print(f"{fw.importances_mean[i]:.3f} +/- {fw.importances_std[i]:.3f}")

feature = [s[0] + m for s in side for m in measurements]
feature = feature + list(demo_var.columns)
plt.barh(feature, fw.importances_mean)
plt.xlabel("Permutation Importance")
plt.show()


# ----- Calculate patameters weight for linear model -----
linear_svm = svm.SVC(kernel = 'linear', random_state = 0)
clf = linear_svm.fit(x_train, y_train)
print('Weight of the Linear model: ', clf.coef_)
plt.barh(feature, abs(clf.coef_[0]))
plt.xlabel("Feature weight for Linear model")
plt.show()
print('accuracy of the Linear model: ', metrics.accuracy_score(y_test,clf.predict(x_test)))

end_time = time.time()
print("The run time of the code is %f seconds" %(end_time - start_time))
