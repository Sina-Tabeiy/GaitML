# --------------- INTRO ---------------

# This version is working based on v0.0, The modifications involve:
# - Adding mean value of kinematics.
# - Adding StandardScaler and MinMaxScaler (Normalizer).
# - Adding Bayesian Optimization to tune the hyperparameters.
# - Adding coefficient matrix to get the feature importance for the linear model.
# - Adding SHAP.

# --------------- NOTE ---------------
# The code works way better while having the legs seprated.
# ------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics, svm
from bayes_opt import BayesianOptimization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import skewtest, kurtosistest
import sys
import time
import shap
sys.path.append('D:\Sina Tabeiy\Project\Functions')
import featurextractor


# --------------- Load Data ---------------

# ----- Define variables -----
file_directory = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Pre Data'
# measurements = ['pctToeOff', 'pctSimpleAppuie', 'distPas', 'distFoulee', 'tempsFoulee']
measurements = ['angAtFullCycle','pctToeOff', 'pctSimpleAppuie', 'distPas', 'distFoulee', 'tempsFoulee']
# joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']
joint_name = ['Hip', 'Knee', 'Ankle']
side = ['Right', 'Left']
output_dir = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\CSV Output'
significance_value = -1

# ----- Extract variables from .mat files -----
# The output data will be in the order of all pre intervention files and then all post intervention files. 
# Since the folder only contains Pre data so we are gonna have the pre data only. 
all_data = featurextractor.mean_feature_extractor(file_directory, measurements, output_dir, separate_legs = True, joint_names = joint_name)
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
feature = [j + axis for j in joint_name for axis in ['x', 'y', 'z']] + [ m for m in measurements[1:]] # If separate_legs = True
feature = feature + list(demo_var.columns)
sktst = skewtest(all_data) #short form of skewtest
print("Skewness values:\n", sktst.statistic)
print("Skewness test p-values:\n ", sktst.pvalue)
kurtst = kurtosistest(all_data) #short form of kurtosis test
print("Kurtosis values:\n", kurtst.statistic)
print("Kurtosis test p-values:\n", kurtst.pvalue)


# ----- Load the final result and label the data -----
gps = pd.read_csv(r'D:\Sina Tabeiy\Project\Results\GPS_results_separatedlegs\GPS_output.csv', index_col=False)
gps.drop(columns=['Unnamed: 0'], inplace=True)
diffrence = np.diff(gps,axis=1)
labels = np.where(diffrence < significance_value, 1, 0).flatten()

# *******ACTION REQUIRED*********: IF RUNNING WITH separate_legs = False with the file which consider legs together
# DEACTIVATE THE FOLLOWING LINE. OTHERWISE, KEEP IT ACTIVATED.
# labels = np.repeat(labels, 2)

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
    return acc.mean()

print('************************************')
print('Bayesian Optimization initiated.')
# kernel_names = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_names = ['rbf']

pbounds = {
            'kernel_index' : (0, len(kernel_names)-1),
            'C': (1,1000),
            'gamma': (0.001, 1),
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
y_hat = SVM_best.predict(x_test)
print("Test set score: ", SVM_best.score(x_test, y_test))
print(metrics.classification_report(y_test, y_hat))

# ----- Confusion Matrix -----
labels = ['Non-responder', 'Responder']
confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_hat)
disp = ConfusionMatrixDisplay(confusion_mat, display_labels=labels)
disp.plot()
plt.show()

# ----- Calculate SHAP values for the best non-linear model -----
explainer = shap.KernelExplainer(SVM_best.predict, x_train)
shap_values = explainer.shap_values(x_test)




# feature = [s[0] + m for s in side for m in measurements] # If separate_legs = False
feature = [j + axis for j in joint_name for axis in ['x', 'y', 'z']] + [ m for m in measurements[1:]] # If separate_legs = True
feature = feature + list(demo_var.columns)

# Plot SHAP values
shap.summary_plot(shap_values, x_test, feature_names=feature)

# ----- Calculate parameters weight for linear model -----
linear_svm = svm.SVC(kernel = 'linear', random_state = 0)
clf = linear_svm.fit(x_train, y_train)
print('Weight of the Linear model: ', clf.coef_)
plt.barh(feature, abs(clf.coef_[0]))
plt.xlabel("Feature weight for Linear model")
plt.show()
print('accuracy of the Linear model: ', metrics.accuracy_score(y_test, clf.predict(x_test)))

end_time = time.time()
print("The run time of the code is %f seconds" % (end_time - start_time))
