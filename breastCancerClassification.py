# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:51:05 2024

@author: NAMAN PURI
"""
# do you want to do multiclass single label classification on specific type of breast cancer
# do you want to do binary classification if the cancer relapsed or not.

# count of deceased and alive, count of each detailed breast cancer type in the dataset


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Source: https://www.cbioportal.org/study/summary?id=brca_metabric
df = pd.read_table("brca_metabric_clinical_data.tsv")
print(df)

#Data Cleaning

#1. dropping the following 14 columns which provide ancillary or repetitive information:
columnsToBeDropped = ['Study ID', 'Patient ID', 'Sample ID', 'Cancer Type','Cohort', 'Number of Samples Per Patient', 'Sample Type', "Patient's Vital Status", 'Pam50 + Claudin-low subtype', 'ER status measured by IHC', 'HER2 status measured by SNP6', 'Tumor Other Histologic Subtype', 'Oncotree Code', '3-Gene classifier subtype', 'Sex']
df.drop(columnsToBeDropped, axis = 1, inplace=True)
print(df)

print("Non-Null Count + Data Type for each attribute: ")
print(df.info())

print("Five Point Summary and more: ", df.describe().T)

# Null Counts for each column
countsOfNaNInEachCol = df.isna().sum()
countsOfNaNInEachCol.plot(kind = 'bar')
plt.xlabel("Attributes")
plt.ylabel("Count of Null Values")
plt.title("Null Count for Each Attribute")
plt.xticks(rotation=90)
plt.savefig("./figuresBreastCancerDetection/countOfNullValuesForEachCol.png", bbox_inches="tight", dpi=300)
plt.show()

# Null Counts for each row
countsOfNaNInEachRow = df.isnull().sum(axis=1).to_numpy()
plt.bar(range(countsOfNaNInEachRow.shape[0]), countsOfNaNInEachRow)
plt.xlabel("Row Index")
plt.ylabel("Count of Null Values")
plt.title("Null Count for Each Row")
plt.savefig("./figuresBreastCancerDetection/countOfNullValuesForEachRow.png", dpi=300)
plt.show()

#2. 'Overall Survival Status' and 'Relapse Free Status' need to be cleaned into int64 data type
def clean_survivalandrelapse_status(status):
    if pd.isna(status):
        return np.nan
    else:
        return int(status.split(':')[0])
    
df['Overall Survival Status'] = df['Overall Survival Status'].apply(clean_survivalandrelapse_status)
df['Relapse Free Status'] = df['Relapse Free Status'].apply(clean_survivalandrelapse_status)

#3. Mapping some ordinal/binary attributes to integers/booleans
mapping_cellularity = {"Low": 1, "Moderate": 2, "High": 3}
df['Cellularity'] = df['Cellularity'].replace(mapping_cellularity)

mapping_chemo_hormone_radio_therapy = {"YES": True, "NO": False}
df['Chemotherapy'] = df['Chemotherapy'].replace(mapping_chemo_hormone_radio_therapy)
df['Hormone Therapy'] = df['Hormone Therapy'].replace(mapping_chemo_hormone_radio_therapy)
df['Radio Therapy'] = df['Radio Therapy'].replace(mapping_chemo_hormone_radio_therapy)

df['Integrative Cluster'] = df['Integrative Cluster'].str.replace(r'^4ER[\+-]$', '4', regex=True)
df['Integrative Cluster'] = df['Integrative Cluster']

# We' ll try to fill the column with NaN values with the mean of that column
# just so that the standard deviation and variance stays fairly the same.
newdf = df.copy()
for i in newdf.columns[newdf.isnull().any(axis=0)]: # applying only on the variables with NaN values
    if newdf[i].dtype != 'object':
        newdf[i].fillna(newdf[i].mean().round(), inplace=True)

countsOfNaNInEachCol_PostCleaning = newdf.isna().sum()
countsOfNaNInEachCol_PostCleaning.plot(kind = 'bar')
plt.xlabel("Attributes")
plt.ylabel("Count of Null Values")
plt.title("Null Count for Each Attribute")
plt.xticks(rotation=90)
plt.savefig("./figuresBreastCancerDetection/countOfNullValuesForEachCol_PostCleaning.png", bbox_inches="tight", dpi=300)
plt.show()

#Dilemma -> Is it better to delete rows with NaN in boolean attributes like Chemotherapy, Hormone Therapy, Radio therapy and Primary Tumor Laterality or take moving average. Cuz the number of null values is simply too large to impute the cells with so much noise presumably
finaldf = newdf[newdf[['Chemotherapy', 'Hormone Therapy', 'Radio Therapy', 'Primary Tumor Laterality']].notna().all(axis=1)]

#integrative cluster coerced to type int
finaldf['Integrative Cluster'] = finaldf['Integrative Cluster'].astype(int)

#Replacing nan values in 'Type of Breast Surgery' with the mode value
mode_string = finaldf['Type of Breast Surgery'].dropna().mode().iloc[0]
finaldf['Type of Breast Surgery'].fillna(mode_string, inplace=True)

#Coercing boolean attributes to int64
boolean_attributes = ['Chemotherapy', 'Hormone Therapy', 'Radio Therapy']
for col in boolean_attributes:
    finaldf[col] = finaldf[col].apply(lambda x: 0 if x==False else 1)

finaldf['Inferred Menopausal State'] = finaldf['Inferred Menopausal State'].apply(lambda x: 0 if x=="Pre" else 1)
finaldf['Primary Tumor Laterality'] = finaldf['Primary Tumor Laterality'].apply(lambda x: 0 if x=="Left" else 1)
posOrneg_status_attributes = ['ER Status', 'HER2 Status', 'PR Status']
for col in posOrneg_status_attributes:
    finaldf[col] = finaldf[col].apply(lambda x: 0 if x=="Negative" else 1)


#Mapping labels to int64
label_counts = finaldf['Cancer Type Detailed'].value_counts()
sorted_labels = label_counts.index.tolist()
label_to_factorized = {label: factor for factor, label in enumerate(sorted_labels)}
finaldf['Cancer Type Detailed Factorized'] = finaldf['Cancer Type Detailed'].map(label_to_factorized)

countsOfNaNInEachCol_PostRemoving = finaldf.isna().sum()
countsOfNaNInEachCol_PostRemoving.plot(kind = 'bar')
plt.xlabel("Attributes")
plt.ylabel("Count of Null Values")
plt.title("Null Count for Each Attribute")
plt.xticks(rotation=90)
plt.savefig("./figuresBreastCancerDetection/countOfNullValuesForEachCol_PostRemoving.png", bbox_inches="tight", dpi=300)
plt.show()

# Summary:
#  1. We removed all ancillary or repetitive attributes
#  2. Fixed the attributes in wrong formats
#  3. Then we dropped the rows for columns having nan count>500
#  4. This lead us to a clean data with no nan values after replacing a 11 sample points in 'Type of Breast Surgery' with its mode

# starting DF: 2509 X 39
# resulting DF: 1870 X 25
# result: 0 NaNs accross the dataset

# EDA

#1. Value Counts for each label
valueCountsOfCancerTypes = finaldf['Cancer Type Detailed'].value_counts()
print(valueCountsOfCancerTypes)

logFrequencies = np.log(valueCountsOfCancerTypes)

countLabelsFig = plt.figure(figsize = (10,5))

plt.bar(range(len(logFrequencies)), logFrequencies.values)
plt.xlabel("Sorted Class Index")
# plt.xticks(rotation=90)
plt.ylabel("Log frequency of samples")
plt.title("Count of Detailed Cancer Types")
plt.savefig("./figuresBreastCancerDetection/cnt_samples.png", bbox_inches="tight", dpi=300)
plt.show()

# Inference 1: We can make out that the distribution among these 6 classes is Long-Tailed

df0 = finaldf.copy()
df0.drop(['Type of Breast Surgery','Cancer Type Detailed'], inplace=True, axis=1)

df1 = finaldf.copy()
df1.drop(['Cancer Type Detailed', 'Cancer Type Detailed Factorized'], inplace = True, axis = 1)
label_counts_surgery = df1['Type of Breast Surgery'].value_counts()
sorted_labels_surgery = label_counts_surgery.index.tolist()
label_surgery_to_factorized = {label: factor for factor, label in enumerate(sorted_labels_surgery)}
df1['Type of Breast Surgery Factorized'] = df1['Type of Breast Surgery'].map(label_surgery_to_factorized)
df1.drop(['Type of Breast Surgery'], inplace = True, axis = 1)

plt.bar(label_counts_surgery.index, label_counts_surgery.values)
plt.xlabel("Classes")
# plt.xticks(rotation=90)
plt.ylabel("Frequency of samples")
plt.title("Count of Breast Surgery Samples per Class")
plt.savefig("./figuresBreastCancerDetection/cnt_samples2.png", bbox_inches="tight", dpi=300)
plt.show()

#2. Corr Analysis:
corrMatrix = df1.corr()
sns.heatmap(corrMatrix)
plt.title("Correlation Matrix")
plt.savefig("./figuresBreastCancerDetection/corrMatrix2.png", bbox_inches="tight", dpi=300)    
plt.show()
    
# 3. Training:
# Algorithm:
# 1 - Identify the base model(SVM, Random Forest, XGBoost, MLP)
# 2 - Shuffle the dataset randomly
# 3 - K-fold(10) splitter
# 4 - splitting the data in train and test of 0.25 or 0.2 (optional)	
# 5 - for each fold
# 	- balance the train data using (resampling or reweighting techniques) ### -- general rule of thumb in ML-> first split the dataset into train and test and then apply normalization or other operations to prevent data leaking issues
#   - train the base model on train 
# 	- evaluate the model on holdout set on metrics like acc*, precision, recall, f-measure, roc-auc(it is a good idea to include variance of the skill scores)
# 6 - calculate mean accuracy over 10 folds	
# 7 - Repeat step 4 and 5 for ROS, RUS, No resampling, Focal Loss, CB-Focal Loss, LDAM, Thresholding
# 8 - Pick the best model evaluated using k-fold
# 9 - Fine tune the model given the best resampling or reweighting technique
# 10 - Show the training loss and acc curves, classification report, confusion matrix 
#     and roc-auc value. 

X= df1.drop(['Type of Breast Surgery Factorized'], axis=1)
y = df1['Type of Breast Surgery Factorized'].values

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics, svm
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
skf = KFold(n_splits = 10, random_state=42, shuffle=True)

# 1870/10 = 187 holdout samples and (1870-187) train samples
# and we will have 10 such iterations i.e. the model would be trained and evaluated 10 times

X_array = X.to_numpy()

data_splits_object = skf.split(X_array, y)

predicted_y = []
expected_y = []

for train_index, test_index in data_splits_object:
    x_train, x_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    oversampler = RandomOverSampler(random_state = 42)
    x_train_resampled, y_train_resampled = oversampler.fit_resample(x_train, y_train)
    
    #classifiser Base SVM
    classifier = svm.SVC(kernel='linear', C=1, random_state = 20)
    classifier.fit(x_train_resampled, y_train_resampled)
    
    predictions_for_this_fold = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, predictions_for_this_fold)
    print('Accuracy from this fold is: ' + accuracy.__str__())
    
    predicted_y.extend(predictions_for_this_fold)
    expected_y.extend(y_test)

accuracy_mean = metrics.accuracy_score(expected_y, predicted_y)
print('\n', "Accuracy from all folds is: " + accuracy_mean.__str__())
    
predicted_y_xgb = []
expected_y_xgb = []

for train_index, test_index in skf.split(X_array, y):
    x_train, x_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    oversampler = RandomOverSampler(random_state = 42)
    x_train_resampled, y_train_resampled = oversampler.fit_resample(x_train, y_train)
    
    #classifiser Base XGBoost
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(x_train_resampled, y_train_resampled)
    
    xgb_predictions_for_this_fold = xgb_classifier.predict(x_test)
    xgb_accuracy = metrics.accuracy_score(y_test, xgb_predictions_for_this_fold)
    print('Accuracy from this fold is: ' + xgb_accuracy.__str__())
    
    predicted_y_xgb.extend(xgb_predictions_for_this_fold)
    expected_y_xgb.extend(y_test)

accuracy_mean_xgb = metrics.accuracy_score(expected_y_xgb, predicted_y_xgb)
print('\n', "Accuracy from all folds is: " + accuracy_mean_xgb.__str__())





        










