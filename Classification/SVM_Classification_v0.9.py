# --------------- INTRO ---------------

# This version is working based on v0.0, The modifications involve:
# - Adding mean value of kinematics.
# - Adding Minimum Residuals Maximum Relevance (MRMR) for feature selection.
# - Adding StandardScaler and MinMaxScaler (Normalizer).
# - Adding Bayesian Optimization to tune the hyperparameters.
# - Adding coefficient matrix to get the feature importance for the linear model.
# - Adding SHAP.
# - Changing the labels from GPS to MCID of the 6MWT.

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
import featurExtractor, featureSelection




# --------------- Load Data ---------------

# ----- Define variables -----
file_directory = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Sample'
# example: measurements = ['pctToeOff', 'pctSimpleAppuie', 'distPas', 'distFoulee', 'tempsFoulee']
measurements = ['angAtFullCycle','pctToeOff', 'pctToeOffOppose', 'pctSimpleAppuie',
                'distPas', 'distFoulee', 'tempsFoulee', 'vitFoulee', 'vitCadencePasParMinute']
# example: joint_names = ['Hip', 'Knee', 'Ankle', 'FootProgress', 'Thorax', 'Pelvis']
joint_name = ['Hip', 'Knee', 'Ankle', 'FootProgress']
side = ['Right', 'Left']
output_dir = r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\CSV Output'

# ----- Extract variables from .mat files -----
# The output data will be in the order of all pre intervention files and then all post intervention files. 
# Since the folder only contains Pre data so we are gonna have the pre data only. 
kin_var = featurExtractor.mean_feature_extractor(file_directory, measurements, output_dir, separate_legs = True, joint_names = joint_name)

# ----- Add participants demographic variables -----
demo_var = pd.read_excel(r'D:\Sina Tabeiy\Project\Lokomat Data (matfiles)\Sample\Participants_caracteristic_ML.xlsx')
demo_var.drop(['Patient', 'sex', 'Diagnostique', '6MWT_PRE', '6MWT_POST'], axis = 1, inplace= True)
demo_var.replace(['walker', 'cane', 'none'], [3,2,1], inplace=True)
# ********* ACTION REQUIRED *********:
#           IF RUNNING WITH separate_legs = False, DEACTIVATE THE FOLLOWING LINE.
#           OTHERWISE, KEEP IT ACTIVATED.
demo_var = demo_var.loc[demo_var.index.repeat(2)].reset_index(drop=True)

# ----- Add Label -----
label_prep = demo_var[['GMFCS','delta']]
MCID = []
# Since Python consider a range of a to b-1 while having range(a,b), I added 1 the maximum range og each GMFCS level.
GMFCS_MCID = {1: range(4,29), 2: range(4,29), 3: range(9,20), 4: range(10,28)} 

for i in range(len(label_prep['GMFCS'])):

    # if  (min(GMFCS_MCID[label_prep['GMFCS'][i]]) <= label_prep['delta'][i]) and (label_prep['delta'][i] <= max(GMFCS_MCID[label_prep['GMFCS'][i]])):
    #     MCID.append(1)
    if label_prep['delta'][i] >= max(GMFCS_MCID[label_prep['GMFCS'][i]]):
        MCID.append(1)
    else:
        MCID.append(0)
MCID = pd.Series(MCID)

# ----- Add all features to each other -----
demo_var.drop(['delta'], axis = 1, inplace= True)
all_data = pd.concat((kin_var, demo_var), axis=1)

# ----- Check the normality of features -----
sktst = skewtest(all_data.values)
print("Skewness values:\n", sktst.statistic)
print("Skewness test p-values:\n ", sktst.pvalue)
kurtst = kurtosistest(all_data.values)
print("Kurtosis values:\n", kurtst.statistic)
print("Kurtosis test p-values:\n", kurtst.pvalue)

# ********* ACTION REQUIRED *********: 
#           IF RUNNING WITH separate_legs = False with the file which consider legs together
#           DEACTIVATE THE FOLLOWING LINE. OTHERWISE, KEEP IT ACTIVATED.
# labels = np.repeat(labels, 2)

# --------------- ML algorithm: Support Vector Machine (SVM) ---------------

# ----- Select features ------
selectedFeatures= featureSelection.selector(allfeatures=all_data, label=MCID, number_of_important_features= 10)
selected_data = all_data[selectedFeatures]
selected_data = selected_data.values

# ----- Train / test Split -----
x_train, x_test, y_train, y_test = train_test_split(selected_data, MCID, test_size = 0.25, random_state = 0)

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
kernel_names = ['poly']

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

# Plot SHAP values
shap.summary_plot(shap_values, x_test, feature_names=selectedFeatures)