from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import operator
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#from matplotlib import style
#style.use("ggplot")
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as sio

file=sio.loadmat('PermutedData.mat')
dataset=file['X']

X= dataset[:,[0,1,2,4,5]]  #[0,1,2,3,4,5]

Y=dataset[:,6]

X_train = X[:217,:]
X_v = X[217:267,:]
X_test=X[267:310,:]

Y_train = Y[:217]
Y_v = Y[217:267]
Y_test = Y[267:310]
X_scaled = preprocessing.scale(X_train)
X_scaled_v=preprocessing.scale(X_v)
X_scaled_t=preprocessing.scale(X_test)

def error(y1,y2):
	class_right=np.count_nonzero(y1==y2)
	mistakes=len(y1)-class_right
	error = (mistakes/len(y1))
	return (error)

#loop over C
#loop over different kinds of kernels since we do not know if our data is linearly separable
C_param = np.arange(0.1, 2.1, 0.1)
# [0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2]
kernel = ['rbf']#,'rbf','poly']
result=[]
result1=[]
result2=[]
for c in C_param:
    for k in kernel:
        clf = svm.SVC( C = c, class_weight='balanced', kernel=k)
        clf.fit(X_scaled,Y_train)
        y_v_p= clf.predict(X_scaled_v)
        err = error(Y_v,y_v_p)
        result.append(err)
        result1.append(c)
        result2.append(k)

min_index,min_value = min(enumerate(result),key=operator.itemgetter(1))
min1=result[min_index]
min2=result1[min_index]
min3=result2[min_index]

clf1 = svm.SVC( C = min2, class_weight='balanced', kernel=min3)
clf1.fit(X_scaled,Y_train)
y_t_p= clf1.predict(X_scaled_t)
err1 = error(Y_test,y_t_p)

print ('The smallest test error is {}%, resulting from C={} and a {} kernel.'.
       format(100*err1, result1[min_index], result2[min_index]))
print(result, result1, result2)
#print(clf.support_vectors_) #prints all the support vectors
#print(clf.support_) #indices of support vectors
#print(clf.n_support_) #number of support vectors for each class