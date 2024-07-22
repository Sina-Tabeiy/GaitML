# --------------- INTRO ---------------

# This version is working based on v0.0, The modifications involve:
# - Adding StandardScaler and MinMaxScaler (Normalizer).
# - Adding Bayesian Optimization to tune the hyperparameters.
# - Adding coefficient matrix to get the feature importance for the linear model.
# - Adding Permutation importance.

# --------------- NOTE ---------------
# The code works way better while having the legs seprated.
# Maximum acc of 91% is achieved in kernel = POLY.
# ------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn import metrics, svm
from bayes_opt import BayesianOptimization
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
all_data = featurextractor.feature_extractor(file_directory, measurements, output_dir, separate_legs = True)
if isinstance(all_data, pd.DataFrame):
    all_data = all_data.values

# ----- Add participants demographic variables -----
demo_var = pd.read_excel(r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Pre Data\Participants_caracteristic_ML.xlsx')
demo_var.drop(['Patient', 'sex','masse', 'taille', 'Diagnostique'], axis = 1, inplace= True)
demo_var.replace(['walker', 'cane', 'none'], [3,2,1], inplace=True)
demo_val = demo_var.values
# *******ACTION REQUIRED*********: IF RUNNING WITH separate_legs = False, DEACTIVATE THE FOLLOWING LINE. OTHERWISE, KEEP IT ACTIVATED.
demo_val = np.repeat(demo_val, 2, axis = 0)

all_data = np.concatenate((all_data, demo_val), axis=1)

# ----- Load the final result and label the data -----
gps = pd.read_csv(r'D:\Sina Tabeiy\Project\Results\GPS_results_all\GPS_output.csv')
diffrence = np.diff(gps,axis=1)
labels = np.where(diffrence < significance_value, 1, 0).flatten()

# *******ACTION REQUIRED*********: IF RUNNING WITH separate_legs = False, DEACTIVATE THE FOLLOWING LINE. OTHERWISE, KEEP IT ACTIVATED.
labels = np.repeat(labels, 2)

# --------------- ML algorithm: Support Vector Machine (SVM) ---------------
start_time = time.time()

# ----- Train / test Split -----
x_train, x_test, y_train, y_test = train_test_split(all_data, labels, test_size = 0.25, random_state = 0)

# ----- Scale the data -----
normalizer = MinMaxScaler()
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ----- Bayesian Optimization -----

def svm_model(C, gamma,kernel_index):
    kernel = kernel_names[int(kernel_index)]
    SVM = svm.SVC(kernel=kernel, C=C, gamma=gamma, random_state=0)
    acc = cross_val_score(SVM, x_train, y_train, cv=5)
    # SVM.fit(x_train,y_train)
    # y_hat = SVM.predict(x_test)
    # acc = metrics.accuracy_score(y_test, y_hat)
    return acc.mean()

print('************************************')
print('Bayesian Optimization initiated.')

# kernel_names = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_names = ['poly']
pbounds = {
            'kernel_index' : (0, len(kernel_names)-1),
            'C': (1,1000),
            'gamma': (0.001, 1),
            # 'degree': (2,5)  # Only relevant for 'poly' kernel
            # 'coef0': [0.0, 0.1, 0.5, 1.0]  # Only relevant for 'poly' and 'sigmoid' kernels
            }

optimizer = BayesianOptimization( f = svm_model, pbounds = pbounds, random_state=0, verbose=3)
optimizer.maximize(init_points=5, n_iter=50)
best_parameters = optimizer.max['params']
best_parameters['C'] = float(best_parameters['C'])

print("********** Best parameters ********** ")
print("The best parameters are: ", optimizer.max)
print(kernel_names[int(best_parameters['kernel_index'])], best_parameters['C'], best_parameters['gamma'])
SVM_best = svm.SVC(kernel=kernel_names[int(best_parameters['kernel_index'])], C=best_parameters['C'], gamma=best_parameters['gamma'], random_state=0)
SVM_best.fit(x_train, y_train)
print("Test set score: ", SVM_best.score(x_test, y_test))
print(metrics.classification_report(y_test, SVM_best.predict(x_test)))

# # ----- Calculate parameters weight for the best non-linear model -----
fw = permutation_importance(SVM_best, x_test, y_test, n_repeats = 20, n_jobs=-1, random_state = 0)
for i in range(len(fw.importances_mean)):
        print(f"{fw.importances_mean[i]:.3f} +/- {fw.importances_std[i]:.3f}")

# feature = [s[0] + m for s in side for m in measurements]
feature = [m for m in measurements]
feature = feature + list(demo_var.columns)
plt.barh(feature, fw.importances_mean)
plt.xlabel("Permutation Importance")
plt.show()


# # ----- Calculate patameters weight for linear model -----
# linear_svm = svm.SVC(kernel = 'linear', random_state = 0)
# clf = linear_svm.fit(x_train, y_train)
# print('Weight of the Linear model: ', clf.coef_)
# plt.barh(feature, abs(clf.coef_[0]))
# plt.xlabel("Feature weight for Linear model")
# plt.show()
# print('accuracy of the Linear model: ', metrics.accuracy_score(y_test,clf.predict(x_test)))

# end_time = time.time()
# print("The run time of the code is %f seconds" %(end_time - start_time))
