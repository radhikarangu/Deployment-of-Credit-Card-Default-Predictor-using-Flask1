# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:26:19 2021

@author: RADHIKA
"""

import numpy as np #numerical computation
import pandas as pd #data wrangling
import matplotlib.pyplot as plt #plotting package
#Next line helps with rendering plots
%matplotlib inline
import matplotlib as mpl #add'l plotting functionality
mpl.rcParams['figure.dpi'] = 400 #high res figures

url='https://raw.githubusercontent.com/Technocolabs100/Project-Data-Set-Repository/master/Data%20set/default_of_credit_card_clients.xls'
df = pd.read_excel(url, error_bad_lines=False)
df.head()
df_zero_mask = df == 0
feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)
sum(feature_zero_mask)
# 315
df_clean = df.loc[~feature_zero_mask,:].copy()
df_clean.shape
df_clean['ID'].nunique()
# 29685
df_clean['EDUCATION'].value_counts()
# 2    13884
# 1    10474
# 3     4867
# 5      275
# 4      122
# 6       49
# 0       14
# Name: EDUCATION, dtype: int64
df_clean['EDUCATION'].replace(to_replace=[0, 5, 6], value=4, inplace=True)
df_clean['EDUCATION'].value_counts()
# 2    13884
# 1    10474
# 3     4867
# 4      460
# Name: EDUCATION, dtype: int64
df_clean['MARRIAGE'].value_counts()
# 2    15810
# 1    13503
# 3      318
# 0       54
# Name: MARRIAGE, dtype: int64
#Should only be (1 = married; 2 = single; 3 = others).
df_clean['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)
df_clean['MARRIAGE'].value_counts()
# 2    15810
# 1    13503
# 3      372
# Name: MARRIAGE, dtype: int64
df_clean['PAY_1'].value_counts()
# 0                13087
# -1                5047
# 1                 3261
# Not available     3021
# -2                2476
# 2                 2378
# 3                  292
# 4                   63
# 5                   23
# 8                   17
# 6                   11
# 7                    9
# Name: PAY_1, dtype: int64
missing_pay_1_mask = df_clean['PAY_1'] == 'Not available'
sum(missing_pay_1_mask)
# 3021
df_missing_pay_1 = df_clean.loc[missing_pay_1_mask,:].copy()
df_missing_pay_1.shape
# (3021, 25)
df_missing_pay_1['PAY_1'].head(3)
df_missing_pay_1['PAY_1'].value_counts()
df_missing_pay_1.columns
url='https://raw.githubusercontent.com/Technocolabs100/Project-Data-Set-Repository/master/Data%20set/cleaned_data.csv'
df = pd.read_csv(url, error_bad_lines=False)
df.head()
df.columns
features_response = df.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','EDUCATION_CAT', 'graduate school', 'high school', 'none','others', 'university']
features_response = [item for item in features_response if item not in items_to_remove]
features_response
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(df[features_response[:-1]].values, df['default payment next month'].values,
test_size=0.2, random_state=24)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# (21331, 17)
# (5333, 17)
# (21331,)
# (5333,)
df_missing_pay_1.shape
features_response[4]
np.median(X_train[:,4])
np.random.seed(seed=1)
fill_values = [0, np.random.choice(X_train[:,4], size=(3021,), replace=True)]
fill_strategy = ['mode', 'random']
fill_values[-1]

fig, axs = plt.subplots(1,2, figsize=(8,3))
bin_edges = np.arange(-2,9)
axs[0].hist(X_train[:,4], bins=bin_edges, align='left')
axs[0].set_xticks(bin_edges)
axs[0].set_title('Non-missing values of PAY_1')
axs[1].hist(fill_values[-1], bins=bin_edges, align='left')
axs[1].set_xticks(bin_edges)
axs[1].set_title('Random selection for imputation')
plt.tight_layout()

from sklearn.model_selection import KFold
k_folds = KFold(n_splits=4, shuffle=True, random_state=1)
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

for counter in range(len(fill_values)):
    #Copy the data frame with missing PAY_1 and assign imputed values
    df_fill_pay_1_filled = df_missing_pay_1.copy()
    df_fill_pay_1_filled['PAY_1'] = fill_values[counter]
    
    #Split imputed data in to training and testing, using the same
    #80/20 split we have used for the data with non-missing PAY_1
    X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
    train_test_split(
        df_fill_pay_1_filled[features_response[:-1]].values,
        df_fill_pay_1_filled['default payment next month'].values,
    test_size=0.2, random_state=24)
    
    #Concatenate the imputed data with the array of non-missing data
    X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
    y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
    
    #Use the KFolds splitter and the random forest model to get
    #4-fold cross-validation scores for both imputation methods
    imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')
    
    test_score = imputation_compare_cv['test_score']
    print(fill_strategy[counter] + ' imputation: ' +
          'mean testing score ' + str(np.mean(test_score)) +
          ', std ' + str(np.std(test_score)))
    
pay_1_df = df.copy()
features_for_imputation = pay_1_df.columns.tolist()
items_to_remove_2 = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','EDUCATION_CAT', 'graduate school', 'high school', 'none','others', 'university', 'default payment next month', 'PAY_1']
features_for_imputation = [item for item in features_for_imputation if item not in items_to_remove_2]
features_for_imputation
X_impute_train, X_impute_test, y_impute_train, y_impute_test = \
train_test_split(
    pay_1_df[features_for_imputation].values,
    pay_1_df['PAY_1'].values,
test_size=0.2, random_state=24)
rf_impute_params = {'max_depth':[3, 6, 9, 12],
             'n_estimators':[10, 50, 100, 200]}
from sklearn.model_selection import GridSearchCV
cv_rf_impute = GridSearchCV(rf, param_grid=rf_impute_params, scoring='accuracy',n_jobs=-1, iid=False, refit=True,cv=4, verbose=2, error_score=np.nan, return_train_score=True)
cv_rf_impute.fit(X_impute_train, y_impute_train)
impute_df = pd.DataFrame(cv_rf_impute.cv_results_)
impute_df
cv_rf_impute.best_params_
# {'max_depth': 12, 'n_estimators': 100}
cv_rf_impute.best_score_
# 0.7337676389523727
pay_1_value_counts = pay_1_df['PAY_1'].value_counts().sort_index()
pay_1_value_counts
pay_1_value_counts/pay_1_value_counts.sum()
y_impute_predict = cv_rf_impute.predict(X_impute_test)
from sklearn import metrics
metrics.accuracy_score(y_impute_test, y_impute_predict)

fig, axs = plt.subplots(1,2, figsize=(8,3))
axs[0].hist(y_impute_test, bins=bin_edges, align='left')
axs[0].set_xticks(bin_edges)
axs[0].set_title('Non-missing values of PAY_1')
axs[1].hist(y_impute_predict, bins=bin_edges, align='left')
axs[1].set_xticks(bin_edges)
axs[1].set_title('Model-based imputation')
plt.tight_layout()
X_impute_all = pay_1_df[features_for_imputation].values
y_impute_all = pay_1_df['PAY_1'].values
rf_impute = RandomForestClassifier(n_estimators=100, max_depth=12)
rf_impute
rf_impute.fit(X_impute_all, y_impute_all)
df_fill_pay_1_model = df_missing_pay_1.copy()
df_fill_pay_1_model['PAY_1'].head()
df_fill_pay_1_model['PAY_1'].value_counts().sort_index()
X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)
print(X_fill_pay_1_train.shape)
print(X_fill_pay_1_test.shape)
print(y_fill_pay_1_train.shape)
print(y_fill_pay_1_test.shape)
X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
print(X_train_all.shape)
print(y_train_all.shape)
rf
imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')
imputation_compare_cv['test_score']
# array([0.76890992, 0.77309591, 0.77166336, 0.77703366])
np.mean(imputation_compare_cv['test_score'])
# 0.7726757126815554
np.std(imputation_compare_cv['test_score'])
# 0.002931480680760725
df_fill_pay_1_model['PAY_1'] = np.zeros_like(df_fill_pay_1_model['PAY_1'].values)
df_fill_pay_1_model['PAY_1'].unique()
X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)
X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
X_test_all = np.concatenate((X_test, X_fill_pay_1_test), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
y_test_all = np.concatenate((y_test, y_fill_pay_1_test), axis=0)
print(X_train_all.shape)
print(X_test_all.shape)
print(y_train_all.shape)
print(y_test_all.shape)
imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')
np.mean(imputation_compare_cv['test_score'])
rf.fit(X_train_all, y_train_all)
y_test_all_predict_proba = rf.predict_proba(X_test_all)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test_all, y_test_all_predict_proba[:,1])
# 0.7696243835824927
thresholds = np.linspace(0, 1, 101)
df[features_response[:-1]].columns[5]
savings_per_default = np.mean(X_test_all[:, 5])
savings_per_default
# 51601.7433479286
cost_per_counseling = 7500
effectiveness = 0.70
n_pos_pred = np.empty_like(thresholds)
cost_of_all_counselings = np.empty_like(thresholds)
n_true_pos = np.empty_like(thresholds)
savings_of_all_counselings = np.empty_like(thresholds)
counter = 0
for threshold in thresholds:
    pos_pred = y_test_all_predict_proba[:,1]>threshold
    n_pos_pred[counter] = sum(pos_pred)
    cost_of_all_counselings[counter] = n_pos_pred[counter] * cost_per_counseling
    true_pos = pos_pred & y_test_all.astype(bool)
    n_true_pos[counter] = sum(true_pos)
    savings_of_all_counselings[counter] = n_true_pos[counter] * savings_per_default * effectiveness
    
    counter += 1
net_savings = savings_of_all_counselings - cost_of_all_counselings 
# plt.plot(thresholds, cost_of_all_counselings)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
plt.plot(thresholds, cost_of_all_counselings)
plt.xlabel('Threshold')
plt.ylabel('cost_of_all_counselings')
plt.xticks(np.linspace(0,1,11))
plt.grid(True)

# plt.plot(thresholds, savings_of_all_counselings)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
plt.plot(thresholds, savings_of_all_counselings)
plt.xlabel('Threshold')
plt.ylabel('savings_of_all_counselings')
plt.xticks(np.linspace(0,1,11))
plt.grid(True)

mpl.rcParams['figure.dpi'] = 400
plt.plot(thresholds, net_savings)
plt.xlabel('Threshold')
plt.ylabel('Net savings (NT$)')
plt.xticks(np.linspace(0,1,11))
plt.grid(True)

max_savings_ix = np.argmax(net_savings)
thresholds[max_savings_ix]
# 0.2
net_savings[max_savings_ix]
# 15446325.35991916

cost_of_defaults=sum(y_test_all)*savings_per_default
cost_of_defaults
net_savings[max_savings_ix]/(cost_of_defaults)
net_savings[max_savings_ix]/len(y_test_all)

plt.plot(cost_of_all_counselings/len(y_test_all),net_savings/len(y_test_all))
plt.xlabel('cost of counselling per Account')
plt.ylabel('Net savings per Account (NT$)')

plt.plot(thresholds, n_pos_pred/len(y_test_all))
plt.ylabel('Flag rate')
plt.xlabel('Threshold')

plt.plot(n_true_pos/sum(y_test_all), np.divide(n_true_pos, n_pos_pred))
plt.xlabel('Recall')
plt.ylabel('Precision')


plt.plot(thresholds, np.divide(n_true_pos, n_pos_pred), label='Precision')
plt.plot(thresholds, n_true_pos/sum(y_test_all), label='Recall')
plt.xlabel('Threshold')
plt.legend()


import pickle
pickle.dump(rf,open('final_model.pkl','wb'))
