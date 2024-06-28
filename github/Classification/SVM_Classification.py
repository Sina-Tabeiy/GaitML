import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, svm
import os
import sys
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
all_data = featurextractor.feature_extractor(file_directory, measurements, output_dir)
all_data = np.asanyarray(all_data)
# ----- Load the final result and label the data -----
gps = pd.read_csv(r'D:\Sina Tabeiy\Project\Clustering\Results\GPS_results\GPS_output.csv')
diffrence = np.diff(gps,axis=1)
labels = np.where(diffrence < significance_value, 1, 0).flatten()
"""
labeled_data = np.concatenate((all_data, labels), axis = 1)
pd.DataFrame(labeled_data).to_csv(output_dir + r'\labeled_data.csv', header = False, index = False)
"""
# ----------

# --------------- ML algorithm: Support Vector Machine (SVM) ---------------

# ----- Train/Test Data split -----
x_train, x_test, y_train, y_test = train_test_split(all_data, labels, test_size = 0.3, random_state = 0)

# ----- Create model -----
SVM = svm.SVC(kernel = 'rbf')
SVM.fit(x_train, y_train)

# ----- Test model -----
y_hat = SVM.predict(x_test)
acc = metrics.accuracy_score(y_test, y_hat)
print('************************************')
print("The accuracy of the model is: %d" %acc)
print('************************************')


