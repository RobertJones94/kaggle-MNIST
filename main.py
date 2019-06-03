# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:15:06 2019

@author: Rob

Kaggle MNIST data set

https://www.kaggle.com/c/digit-recognizer

# going to try KNN, SVM and NN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotFunctions import MNIST_Plot
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn import svm
filePath = 'D:\\Rob\\Documents\\DataScience\\Kaggle\\MNIST/'
train = pd.read_csv(filePath + 'train.csv')
test = pd.read_csv(filePath + 'test.csv')

# let's see what we're dealing with here
for i in range(10):
    MNIST_Plot(train.iloc[i, 1:])
    
# counts of each number
counts = train['label'].value_counts() 

# check missing values
pd.isna(train).sum().sum() # none

# have a look at the distributions
# slightly less 5's
counts.sort_index().plot(kind='bar')

train.max().max() # range between 0 and 255

#first job is to rescale between 0 and 1
X_train = train.iloc[:, 1:]/255
y_train = train.iloc[:, 0]

# still all good!
MNIST_Plot(X_train.iloc[0,:])


############# 
# model time
#
# SVM
# KNN
# NN
#

###33
# svm is very slow - PCA first


pcaTrain = PCA().fit(X_train)

plt.title('princomps')
pcaVarExp = np.cumsum(pcaTrain.explained_variance_ratio_)
cutoff= 0.9
plt.plot(pcaVarExp)
plt.plot([0,800],[cutoff,cutoff])
plt.xlabel('Components')
plt.ylabel('Variance Explained')

# only get the first 90% of the variance
X_pca = PCA(n_components= np.argmax(pcaVarExp > cutoff)).fit(X_train)
X_pcaTrain = X_pca.transform(X_train)
# turned 784 into just 86


clf = svm.SVC(kernel = 'rbf', gamma = 'auto')

model = clf.fit(X_pcaTrain, y_train)

# just out of curiousity how does it do by itself
preds = model.predict(X_pcaTrain)

confusion_matrix(y_train, preds)
accuracy_score(y_train, preds)

# Cross validation
score = cross_val_score(clf, X_pcaTrain,y_train,cv = 5,scoring='accuracy')
print('Avg Accuracy for i = '+str(i) + ' score = ' + str(np.mean(score))) 

# model fitting
X_pcaTest = X_pca.transform(test/255)
preds = model.predict(X_pcaTest)   

ImageId = np.arange(1,len(preds)+1)

pd.DataFrame([ImageId,preds], index = ['ImageId','Label']).T.to_csv('submission.csv', index=False)
