# IRIS Plant Classification using SVM

from sklearn import datasets

iris = datasets.load_iris()

x= iris.data
y= iris.target

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =\
    train_test_split(x,y,test_size=0.3, random_state=1234, stratify=y)
    
# implement the svm algorithm

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#RBF kernel wiht gamma as 1
svc = SVC(kernel='rbf',gamma=1.0)
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_rbf01 = confusion_matrix(y_test, y_predict)

#RBF kernel with as gamma as 10
svc = SVC(kernel='rbf',gamma=10)
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_rbf10 = confusion_matrix(y_test, y_predict)


#linear kernel
svc = SVC(kernel='linear')
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_linear = confusion_matrix(y_test, y_predict)

#ploynomial kernel

svc = SVC(kernel='poly')
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_poly = confusion_matrix(y_test, y_predict)

#Sigmoid kernel
svc = SVC(kernel='sigmoid')
svc.fit(x_train,y_train)
y_predict = svc.predict(x_test)
cm_sigmoid = confusion_matrix(y_test, y_predict)

