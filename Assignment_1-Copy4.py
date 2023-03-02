#!/usr/bin/env python
# coding: utf-8

# ## 0. Some Packages

# In[1]:


# 0.1 k-nearest neighbor imputer for imputing missing data
from sklearn.impute import KNNImputer  

# 0.2 feature selection: linear regression, logistic regression (foreward and backward selection), and random forest for feature importance
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector as SFS 
from sklearn.ensemble import RandomForestClassifier 

# 0.3 building models
from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree  # decision tree for selecting complex features
from sklearn.preprocessing import StandardScaler  # standard scaler for scaling the vairables to avoid bias while doing the imputation
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV  # model selection

# 0.4 model evaluation
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score, confusion_matrix  # evaluation metrics

# 0.5 other packages
import pandas as pd    # process tabular data
import numpy as np     # deal with math operations
import matplotlib.pyplot as plt  # make figures


# ## 1. Data Preparation 

# In[2]:


# 1.0 load the data
df = pd.read_csv('Downloads/Assignment_1_data.csv')  # load data

# 1.1 do some the data overview to see column types
df.head(10)


# In[3]:


# 1.2 convert the categorical values or boolean values into numerical values
df['gender'] = df['gender'].replace({'F':0, 'M':1})
df['outcome'] = df['outcome'].replace({'False':0, 'True':1}).astype('int64')

# 1.3 see the missing overview for the columns
df.isnull().sum()


# In[4]:


# 1.4 deal with columns

# 1.4.1 find the columns with more than 55% missing values
numRows = df.shape[0]
dropCol = []
for col in df.columns:
    misPercent = (df[col].isnull().sum())/numRows  # percent of missing
    if misPercent > 0.55:
        dropCol.append(col)
        
# 1.4.2 modify the current data frame by dropping columns
df.drop(dropCol, axis=1, inplace=True)  


# In[5]:


# 1.5 deal with rows

# 1.5.1 find the rows with more than 55% missing values
numCols = df.shape[1]
dropRow = []
for i, row in df.iterrows():
    misPercent = (row.isnull().sum())/numRows
    if misPercent > 0.55:
        dropRow.append(df[i])
        
# 1.5.2 modify the current data frame by dropping rows
df.drop(dropRow, axis=0, inplace=True)


# In[6]:


# 1.6 Impute columns with <55% missing values

# 1.6.1 create the imputer, using 5 nearest neighbors
Imputer = KNNImputer(n_neighbors=5)  

# 1.6.2 store the columns with missing values again in MisCol
MisCol = []    
for col in df.columns:
    if df[col].isnull().sum() > 0:
        MisCol.append(col)
        
# 1.6.3 store columns without any missing value in OkCol
OkCol = df.drop(columns=MisCol)  

# 1.6.4 get the standard scalar, scaling the vairables to avoid imputation bias
Scaler = StandardScaler()
MisScaled = pd.DataFrame(Scaler.fit_transform(df[MisCol]), columns=MisCol)

# 1.6.5 impute missing columns
MisImputed = Imputer.fit_transform(MisScaled)
MisImputed = pd.DataFrame(MisImputed, columns=MisCol)  # convert to pandas dataframe
dfImputed = pd.concat([MisImputed, OkCol], axis=1)  # concatenate them into one (imputed) df

# 1.6.6 see whether the data frame is imputed
dfImputed.isnull().sum()  


# ## 2. Feature Selection

# In[7]:


# 2.1 split the data into training and testing sets

X = dfImputed.iloc[:, :-1]  # independent variables
y = df['outcome']  # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, shuffle=True, test_size=0.3)


# ### 2.2 Do Feature Selection with Linear Regression

# In[8]:


# 2.2.1 build and fit the linear model
linSel = SelectFromModel(LinearRegression())
linSel = linSel.fit(X_train, y_train)

# 2.2.2 get the coefficients from the fitted linear model (will use their mean as a threshold)
linSel.estimator_.coef_

# 2.2.3 get the columns that were selected by the linear model for feature selection
FeaturesKeptLin = X_train.columns[linSel.get_support()]
print(FeaturesKeptLin)


# In[9]:


# 2.2.4 get new data sets containing only the selected features
linX_train = X_train.loc[:, FeaturesKeptLin]
linX_test = X_test.loc[:, FeaturesKeptLin]


# In[10]:


# 2.2.5 use random forest as the model for "baseline (all features) v.s. after selection" comparision
def RandomForest(X_train, X_test, y_train, y_test):  # build and fit the forest model
    Forest = RandomForestClassifier(n_estimators=500, random_state=100, n_jobs=-1)  # create the forest, use all cpu cores
    Forest = Forest.fit(X_train, y_train)  # fit data into the forest
    
    y_pred = Forest.predict(X_test)  # get the predicted value of y
    print('Average Precision:', average_precision_score(y_test, y_pred))
    print('AUROC:', roc_auc_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


# In[11]:


# 2.2.6 compute the evaluation after feature selection
print('Linear Regression Coefficient Elimination')
RandomForest(linX_train, linX_test, y_train, y_test)


# ### 2.3 Do Feature Selection with Logistic Regression (Forward and Backward Selection)

# In[12]:


from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.utils import resample

# 2.3.0 the data set is too large that it took forever to compute things out, only select 10

# 2.3.1 build and fit the logistic model
logiModel = LogisticRegression(penalty='l2', C=0.01, solver='liblinear')
logiModel.fit(X_train, y_train)

# 2.3.2 do forward feature selection
FwSel = SFS(logiModel, n_features_to_select=10, direction='forward').fit(X_train, y_train)

# 2.3.3 get the columns that were selected by the logistic model for feature selection
FeaturesKeptFw = FwSel.get_feature_names_out()
FeaturesKeptFw


# In[13]:


# 2.3.4 do backward feature selection
BwSel = SFS(logiModel, n_features_to_select=10, direction='backward').fit(X_train, y_train)

# 2.3.5 get the columns that were selected by the logistic model for feature selection
FeaturesKeptBw = BwSel.get_feature_names_out()
FeaturesKeptBw = FeaturesKeptBw[:-1]  # get rid of the 'outcome'
FeaturesKeptBw


# In[14]:


# 2.3.5 get new data sets containing only the selected features
fwX_train = X_train.loc[:, FeaturesKeptFw]
fwX_test = X_test.loc[:, FeaturesKeptFw]

bwX_train = X_train.loc[:, FeaturesKeptBw]
bwX_test = X_test.loc[:, FeaturesKeptBw]


# In[15]:


# 2.2.6 compare the f1 scores before and after feature selection
print('Logistic Regression Forward Selection')
RandomForest(fwX_train, fwX_test, y_train, y_test)
print('\nLogistic Regression Backward Selection')
RandomForest(bwX_train, bwX_test, y_train, y_test)


# ### 2.4 Do Feature Selection with Logistic Regression (L2 Regularization)

# In[16]:


# 2.4.1 build and fit the logistic model (adjust C manually, keep ≈15 features)
logi2Sel = SelectFromModel(LogisticRegression(penalty='l2', C=0.1, solver='liblinear'))
logi2Sel = logi2Sel.fit(X_train, y_train)

# 2.4.2 get the coefficients from the fitted logistic model and get their mean as a threshold
logi2Sel.estimator_.coef_
np.mean(np.abs(logi2Sel.estimator_.coef_))

# 2.4.3 get the columns that were selected by the logistic model for feature selection
FeaturesKeptLogi2 = X_train.columns[logi2Sel.get_support()]
print(FeaturesKeptLogi2)


# In[17]:


# 2.4.4 get new data sets containing only the selected features
l2X_train = X_train.loc[:, FeaturesKeptLogi2]
l2X_test = X_test.loc[:, FeaturesKeptLogi2]

# 2.4.5 compare the accuracy scores before and after feature selection by calling RandomForest
print('Logistic Regression with L2 Regularization')
RandomForest(l2X_train, l2X_test, y_train, y_test)


# ### 2.5 Do Feature Selection with Logistic Regression (L1 Regularization)

# In[18]:


# 2.5.1 build and fit the logistic model (adjust C manually, keep ≈15 features)
logi1Sel = SelectFromModel(LogisticRegression(penalty='l1', C=0.1, solver='liblinear'))
logi1Sel = logi1Sel.fit(X_train, y_train)

# 2.5.2 get the coefficients from the fitted logistic model and get their mean as a threshold
logi1Sel.estimator_.coef_
np.mean(np.abs(logi1Sel.estimator_.coef_))

# 2.5.3 get the columns that were selected by the logistic model for feature selection
FeaturesKeptLogi1 = X_train.columns[logi1Sel.get_support()]
print(FeaturesKeptLogi1)


# In[19]:


# 2.5.4 get new data sets containing only the selected features
l1X_train = X_train.loc[:, FeaturesKeptLogi1]
l1X_test = X_test.loc[:, FeaturesKeptLogi1]

# 2.5.5 compare the accuracy scores before and after feature selection by calling RandomForest
print('Logistic Regression with L1 Regularization')
RandomForest(l1X_train, l1X_test, y_train, y_test)


# ### 2.6 Find out the Feature Importances Using Random Forest

# In[20]:


# 2.6.1 build and fit the forest model
Forest = RandomForestClassifier(n_estimators=500, random_state=100)  # create the forest
Forest.fit(X_train, y_train)  # fitting data into the forest

# 2.6.2 calculate feature importances
Importances = Forest.feature_importances_  # get the feature importances
Indices = np.argsort(Importances)[::-1]  # sort the importances from the highest to the lowest

# 2.6.3 visualise the importances using a sorted bar chart
plt.ylabel('Feature Importance(0-1)')
plt.bar(range(X_train.shape[1]), Importances[Indices], align='center')  # plot the importance chart

# 2.6.4 plot the columns in a nice evenly-spaced form 
Labels = dfImputed.columns[:-1] # label the columns
plt.xticks(range(X_train.shape[1]), Labels[Indices], rotation=90)  
plt.xlim([-1, X_train.shape[1]])


# In[21]:


# 2.6.5 based on the feature importances, select the top 15 features
FeaturesKeptFores = Labels[Indices[:15]]
print(FeaturesKeptFores)

# 2.6.6 generate new data sets using these 15 features selected
foX_train = X_train.loc[:, FeaturesKeptFores]
foX_test = X_test.loc[:, FeaturesKeptFores]


# In[22]:


# 2.6.7 compute the evaluation after feature selection
print('Random Forest Feature Selection')
RandomForest(foX_train, foX_test, y_train, y_test)


# ### 2.7 Compare the Performances got from Different Combination of Features

# In[23]:


# 2.7.1 baseline model with all features
print('Before')
RandomForest(X_train, X_test, y_train, y_test) 

# 2.7.2 evaluations for different combination of features
print('\nLinear Regression Coefficient Elimination')
RandomForest(linX_train, linX_test, y_train, y_test)
print('\nLogistic Regression Forward Selection')
RandomForest(fwX_train, fwX_test, y_train, y_test)
print('\nLogistic Regression Backward Selection')
RandomForest(bwX_train, bwX_test, y_train, y_test)
print('\nLogistic Regression with L2 Regularization')
RandomForest(l2X_train, l2X_test, y_train, y_test)
print('\nLogistic Regression with L1 Regularization')
RandomForest(l1X_train, l1X_test, y_train, y_test)
print('\nRandom Forest Feature Selection')
RandomForest(foX_train, foX_test, y_train, y_test)


# In[24]:


# 2.7.2 visualize the results in bar charts
labels = ['baseline', 'LinR', 'LogiRF', 'LogiRB', 'LogiL2', 'LogiL1', 'RF']
AP = [0.04072823977494533, 0.04072823977494533, 0.03699305839079946, 0.04332409004983194, 0.04072823977494533, 0.03826856859488109, 0.03580889741481685]

# 2.7.3 create bar chart for averge precision
plt.bar(labels, AP)

# 2.7.4  axis labels and title
plt.xlabel('AP scores')
plt.ylabel('Methods')
plt.title('Average Precision')

plt.show()


# In[25]:


# 2.8.1 visualize the Accuracy score results in bar charts
Accuracy = [0.9643738010413812, 0.9643738010413812, 0.9641911025851831, 0.9643738010413812, 0.9643738010413812, 0.9642824518132822, 0.9641911025851831]

# 2.8.2 create bar chart for Accuracy
plt.bar(labels, Accuracy)

# 2.8.3 put axis labels and title on
plt.xlabel('Accuracy scores')
plt.ylabel('Methods')
plt.title('Accuracy')

plt.show()


# In[26]:


# 2.9.1 visualize the AUROC score results in bar charts
AUROC = [0.5025510204081632, 0.5025510204081632, 0.501228139289823, 0.5062354382776322, 0.5025510204081632, 0.5012755102040817, 0.5]

# 2.9.2 create bar chart for AUROC
plt.bar(labels, AUROC)

# 2.9.3 put axis labels and title on
plt.xlabel('AUROC scores')
plt.ylabel('Methods')
plt.title('AUROC')

plt.show()


# In[27]:


# 2.7.3 feature selected using Logistic Regression (Backward Selection) has the best performance
finX = X.loc[:, FeaturesKeptBw]


# ## 3. Train and Test Models Using Selected Features

# ### 3.1 Do Re-sampling (Under-sampling, Over-sampling) to the Training Sets

# In[28]:


# 3.1.0 import packages
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 3.1.1 see the classification, if look unbalanced then try SMOTE, RandomUnderSampler, and both
y.value_counts()


# In[29]:


# 3.1.2.0 separate data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(finX, y, random_state=100, shuffle=True, test_size=0.3)

# 3.1.2.1 use SMOTE to do overs-ampling over the traning set's minority (True/1) class
Oversam = SMOTE(random_state=100, sampling_strategy='minority')
smoX_train, smoy_train = Oversam.fit_resample(X_train, y_train)

# 3.1.2.2 use RandomUnderSampler to do under-sampling over the majority (False/0) class
Undersam = RandomUnderSampler()
undX_train, undy_train = Undersam.fit_resample(X_train, y_train)

# 3.1.3 see which re-sampling method is the best performing one, using f1 score as a metric
mSMOTE = LogisticRegression(solver='liblinear')
mSMOTE.fit(smoX_train, smoy_train)  # SMOTE
predSMOTE = mSMOTE.predict_proba(X_test)

mUnder = LogisticRegression(solver='liblinear')
mUnder.fit(undX_train, undy_train)  # Under-Sampler
predUnder = mUnder.predict_proba(X_test)

# store prediction, label, abd the method in a nice table
Results = pd.DataFrame({'pred':np.vstack([predSMOTE, predUnder])[:,1], 
                        'label':pd.concat([y_test]*2),
                        'method':['SMOTE']*predSMOTE.shape[0]+['RandomUnder']*predUnder.shape[0]})
Thresh = 0.5

Results['predBi'] = (Results['pred']>Thresh).astype('int')
Results.groupby(['label','method'])['predBi'].value_counts().unstack()

for method in ['SMOTE','RandomUnder']:
    score = f1_score(Results.query('method==@method')['label'], Results.query('method==@method')['predBi'])
    print('F1 score is {:.4f} for method {} '.format(score,method))


# In[30]:


# 3.1.3 select RandomUnderSampler, rename the data sets by storing
balX_train = smoX_train
baly_train = smoy_train

# 3.1.4 see before and after re-sampling
print('Before')
print(y_train.value_counts())
print('\nAfter')
print(baly_train.value_counts())

# we have balX_train (balanced X training set), baly_train (balanced y training set)
# and X_test (X testing set), y_test (y testing set)


# ### 3.2 Train and Test Linear Regression Model

# In[31]:


# 3.2.1 linear regression model is used in predicting continuous variables,
# so it will not be used as a participant in binary prediction


# ### 3.3 Train and Test Logistic Regression Model (Ordinary/with L2/with L1 Regularization)

# #### 3.3.1 Ordinary Logistic Regression

# In[32]:


# 3.3.0 create a function for calculating and printing the AUROC, F1, Precision, and Recall scores out
def Evaluation(modelName, y_test, y_pred):  
    print(modelName)
    print('AUROC:', roc_auc_score(y_test, y_pred))
    print('F1:', f1_score(y_test, y_pred))
    print('AP:', average_precision_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('\n')


# In[33]:


# 3.3.1.0 import pytorch and also the stochastic gradient descent optimizer
import torch
import torch.nn as nn
from torch.optim import SGD

# 3.3.1.1 turn the data into torch data type
torX_train = torch.tensor(balX_train.to_numpy(),dtype=torch.float32)
torX_test = torch.tensor(X_test.to_numpy(),dtype=torch.float32)
m1 = torX_train.shape[0]
m2 = X_test.shape[0]
n = torX_train.shape[1]

tory_train = torch.tensor(baly_train.to_numpy(), dtype=torch.float32).reshape(m1,1)
tory_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(m2,1)

# 3.3.1.2 create a linear layer with n input features and 1 output feature, bias term added
h = torch.nn.Linear(in_features=n, out_features=1, bias=True)

# 3.3.1.3 add the sigmoid function layer for building logistic regression
sigma = torch.nn.Sigmoid()
f = torch.nn.Sequential(h, sigma)

# 3.3.1.4 create a variable from the BCELoss class to calculate binary cross-entropy loss
# the loss during the training phase will be used as a reference in adjusting the learning rate
J_BCE = torch.nn.BCELoss()

# 3.3.1.5 use the Adam optimizer, add some momentum to SGD to help adjust the learning rate
Optimizer = torch.optim.Adam(lr=0.1, params=f.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=50, gamma=0.7)

# 3.3.1.6 train the model
nIter = 600           # number of iterations for training
printInterval = 150    # interval for printing the avg cross-entropy loss

for i in range(nIter):
    Optimizer.zero_grad()  # clear the gradients
    ordy_pred = f(torX_train)  # compute predicted y and calculate the loss
    loss = J_BCE(ordy_pred, tory_train)
    
    loss.backward()  # compute the gradient in the backward pass
    Optimizer.step()  # use it for updating the parameters
    if i == 0 or ((i+1)%printInterval) == 0:  # print the avg BCE loss for intervals 1, 150, 300, ...
        print('Iter {}: average BCE loss is {:.3f}'.format(i+1, loss.item()))

        
# 3.3.1.7 test the model, compute the predicted values using the testing set
with torch.no_grad():
    ordy_pred_test = f(torX_test)  # continuous values

# 3.3.1.8 set a threshold to get binary predictions
Thresh = 0.67  # manually adjust the threshold value based on precision and recall

ordy_pred_test[ordy_pred_test <= Thresh] = 0
ordy_pred_test[ordy_pred_test > Thresh] = 1

Evaluation('\nLogistic Regression (Ordinary)', tory_test, ordy_pred_test)


# #### 3.3.2 Logistic Regression with L2 Regularization

# In[34]:


# 3.3.2.0 use the same training and testing sets as in 3.3.1

# 3.3.2.1 create a linear layer with n input features and 1 output feature, bias term added
h_L2 = torch.nn.Linear(in_features=n, out_features=1, bias=True)

# 3.3.2.2 add the sigmoid function layer for building logistic regression
sigma = torch.nn.Sigmoid()
f_L2 = torch.nn.Sequential(h_L2,sigma)

# 3.3.2.3 use the Adam optimizer, add some momentum to SGD to help adjust the learning rate
Optimizer = torch.optim.Adam(lr = 0.05, params=f_L2.parameters(), weight_decay=0.05)

# 3.3.2.4 train the model
nIter = 600
printInterval = 150

for i in range(nIter):
    Optimizer.zero_grad()  # clear the gradients
    L2y_pred = f_L2(torX_train)  # compute predicted y and calculate the loss
    loss = J_BCE(L2y_pred, tory_train)
    
    loss.backward()  # compute the gradient in the backward pass
    Optimizer.step()  # use it for updating the parameters
    if i == 0 or ((i+1)%printInterval) == 0:  # print the avg BCE loss for intervals 1, 1500, 3000, ...
        print('Iter {}: average BCE loss is {:.3f}'.format(i+1, loss.item()))

        
# 3.3.2.5 test the model, compute the predicted values using the testing set
with torch.no_grad():
    L2y_pred_test = f(torX_test)  # continuous values

# 3.3.2.6 set a threshold to get binary predictions
Thresh = 0.62  # manually adjust the threshold value based on precision and recall

L2y_pred_test[L2y_pred_test <= Thresh] = 0
L2y_pred_test[L2y_pred_test > Thresh] = 1

Evaluation('\nLogistic Regression (L2 Regularization)', tory_test, L2y_pred_test)


# #### 3.3.3 Logistic Regression with L1 Regularization

# In[35]:


# 3.3.3.0 use the same training and testing sets as in 3.3.1

# 3.3.3.1 create a linear layer with n input features and 1 output feature, bias term added
h_L1 = torch.nn.Linear(in_features=n, out_features=1, bias=True)

# 3.3.3.2 add the sigmoid function layer for building logistic regression
sigma = torch.nn.Sigmoid()
f_L1 = torch.nn.Sequential(h_L1,sigma)

# 3.3.3.3 use the Adam optimizer, add some momentum to SGD to help adjust the learning rate
Optimizer = torch.optim.Adam(lr = 0.08, params=f_L1.parameters())

# 3.3.3.4 Define L1 regularization
def L1_reg(model,lbd):
    result = torch.tensor(0)
    for param in model.parameters(): # iterate over all parameters of our model
        result = result + param.abs().sum()
    return lbd*result

nIter = 600
printInterval = 150
lbd = 0.026 # L1 reg strength

for i in range(nIter):
    Optimizer.zero_grad()
    L1y_pred = f_L1(torX_train)
    loss = J_BCE(L1y_pred, tory_train)
    (loss+L1_reg(f_L1, lbd)).backward()
    Optimizer.step()
    if i == 0 or ((i+1)%printInterval) == 0:
        print('Iter {}: average BCE loss is {:.3f}'.format(i+1, loss.item()))

        
# 3.3.2.5 test the model, compute the predicted values using the testing set
with torch.no_grad():
    L1y_pred_test = f(torX_test)  # continuous values

# 3.3.2.6 set a threshold to get binary predictions
Thresh = 0.62  # manually adjust the threshold value based on precision and recall

L1y_pred_test[L1y_pred_test <= Thresh] = 0
L1y_pred_test[L1y_pred_test > Thresh] = 1

Evaluation('\nLogistic Regression (L1 Regularization)', tory_test, L1y_pred_test)


# ### 3.4 Train and Test Decision Tree Model

# In[36]:


# 3.4.1 build a shallow decision tree for a try
treeModel = DTC(criterion='entropy', max_depth=2, random_state=100)
treeModel.fit(balX_train, baly_train)

# 3.4.2 visualize the splitting rules (looks very cool)
plt.figure(figsize=(10,5))
plot_tree(treeModel, filled=True, feature_names=['dbp_mean', 'creatinine_max', 'pt_max', 'ast_min',
       'bilirubin_total_max', 'bilirubin_total_min', 'urineoutput',
       'sofa_liver', 'sofa_cns'],
    class_names=['0','1']
)
plt.show()


# In[37]:


# 3.4.3 define parameters to be tuned for the decision tree model
param = {'max_depth': np.arange(start=1, stop=20, step=1)}  # grid values for try 

# 3.4.4 split data stratifically into folds for cross-validation
stratCV = StratifiedKFold(n_splits=10)  # split into 10 folds where class distribution is balanced

# 3.4.5 build the Decision Tree Model 
treeMod = DTC(criterion='entropy')  # the tree, split criterion is entropy

# 3.4.6 use grid search to select the best max_depth and fit data into it
bestTree = GridSearchCV(treeMod, param_grid=param, scoring='f1', cv=stratCV)  # search for the parameters 
bestTree.fit(balX_train, baly_train)  # which lead to the best f1 score

# 3.4.7 see the best f1 score reached by the best tree
bestTree.best_estimator_
bestTree.best_score_


# In[38]:


# 3.4.4 use the bestTree to do predictions and see its performance
treey_pred = bestTree.predict(X_test)
Evaluation('\nDecision Tree', y_test, treey_pred)


# ### 3.5 Train and Test XGBoost Tree Model

# In[39]:


# 3.5.1 import the XGBoost classifier
from xgboost import XGBClassifier as XGBC

# 3.5.2 define parameters to be tuned for the gradient boosting tree model
param = {
    'n_estimators':np.arange(start=2,stop=20,step=2),  # a range of the number of trees
    'max_depth':np.arange(start=2,stop=6,step=1),  # a range of the depth value for the small trees
    'learning_rate':np.arange(start=0.05,stop=0.4,step=0.05)  # a range of learning rates for adjusting the weight of ensamble
}

# 3.5.3 split data stratifically into folds for cross-validation
stratCV = StratifiedKFold(n_splits=10)

# 3.5.4 find and build the best XGBoost Tree Model through grid search cross-validation
XGBoostMod = XGBC()
bestXGBoost = GridSearchCV(XGBoostMod, param_grid=param, scoring='f1', cv=stratCV, verbose=1,
    n_jobs=-1 # use all cpu cores during grid search
)

# 3.5.5 fit data into the best model
bestXGBoost.fit(balX_train, baly_train)


# In[40]:


# 3.5.6 check and see the hyperparameter values
bestXGBoost.best_params_


# In[41]:


# 3.5.7 check and see the best f1 score
bestXGBoost.best_score_


# In[42]:


# 3.5.8 use the bestXGBoost tree to do predictions and see its performance
XGBoosty_pred = bestXGBoost.predict(X_test)
Evaluation('\nXGBoost Tree', y_test, XGBoosty_pred)


# ### 3.6 Train and Test the Random Forest

# In[43]:


# 3.6.1 define parameters to be tuned
param = {'n_estimators':np.arange(start=2,stop=20,step=2),  # a range of the number of trees
         'max_depth':np.arange(start=2,stop=6,step=1)}  # grid values for try

# 3.6.2 split data stratifically into folds for cross-validation
stratCV = StratifiedKFold(n_splits=8) # split into 8 folds where class distribution is balanced

# 3.6.3 build the Random Forest Model 
rfMod = RandomForestClassifier(criterion='entropy', random_state=100) # the random forest, split criterion is entropy

# 3.6.4 use grid search to select the best hyperparameters and fit data into it
bestRF = GridSearchCV(rfMod, param_grid=param, scoring='f1', cv=stratCV) # search for the parameters
bestRF.fit(balX_train, baly_train) # which lead to the best f1 score

# 3.6.5 see the best f1 score reached by the best random forest
bestRF.best_estimator_
bestRF.best_score_


# In[44]:


# 3.6.6 use the bestRF model to do predictions and see its performance
RFy_pred = bestRF.predict(X_test)
Evaluation('\nRandom Forest', y_test, RFy_pred)


# ### 3.7 Train and Test Support Vector Machine

# In[45]:


# 3.7.1 import support vector classifier
from sklearn.svm import SVC

# 3.7.2 define parameters to be tuned for the support vector classifier
param = {'C':np.arange(start=1,stop=20,step=4)}  # strength of L2 regularization

# 3.7.3 split data stratifically into folds for cross-validation
stratCV = StratifiedKFold(n_splits=10)

# 3.7.4 find and build the best support vector classifier through grid search cross-validation
SVCMod = SVC(kernel='linear')
bestSVC = GridSearchCV(SVCMod, param_grid=param, scoring='f1', cv=stratCV, verbose=1, n_jobs=-1)
bestSVC.fit(balX_train, baly_train)

# 3.7.5 check and see the best hyperparameter values
bestSVC.best_params_


# In[46]:


# 3.7.6 check and see the best f1 score
bestSVC.best_score_


# In[47]:


# 3.6.7 use the bestSVC to do predictions and see its performance
SVCy_pred = bestSVC.predict(X_test)
Evaluation('\nSupport Vector Machine', y_test, SVCy_pred)


# ## 4. Evaluate Model Performances and Select the Best

# In[50]:


# 4.1 check the confusion matrix for each model
logiCM = confusion_matrix(tory_test, ordy_pred_test)
logi2CM = confusion_matrix(tory_test, L2y_pred_test)
logi1CM = confusion_matrix(tory_test, L1y_pred_test)
DeciTreeCM = confusion_matrix(y_test, treey_pred)
XGBoostCM = confusion_matrix(y_test, XGBoosty_pred)
RanForeCM = confusion_matrix(y_test, RFy_pred)
SVMCM = confusion_matrix(y_test, SVCy_pred)

# 4.2 print confusion matrix
print(logiCM, '\n\n', logi2CM, '\n\n', logi1CM, '\n\n', DeciTreeCM, '\n\n', XGBoostCM, '\n\n', RanForeCM, '\n\n', SVMCM)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




