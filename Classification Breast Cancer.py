#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
import matplotlib.pyplot as plt

breast_cancer=load_breast_cancer()
Data=breast_cancer.data
Target=breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(Data, Target, random_state = 0)


def method(choice_1):
    
    if choice_1==1:
        svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
        
        svm_predictions = svm_model_linear.predict(X_test) 
        
        accuracy = svm_model_linear.score(X_test, y_test) 
        
        print("SVM accuracy : ", accuracy*100,'%')
        
        cm = confusion_matrix(y_test, svm_predictions) 
        
    elif choice_1==2:
        
        knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
        
        accuracy = knn.score(X_test, y_test) 
  
        knn_predictions = knn.predict(X_test)
    
        print("KNN accuracy : ", accuracy*100,'%')
        
        cm = confusion_matrix(y_test, knn_predictions) 
        
        
    elif choice_1==3:
        
        gnb = GaussianNB().fit(X_train, y_train) 
        
        gnb_predictions = gnb.predict(X_test) 
        
        accuracy = gnb.score(X_test, y_test)
        
        print("Naive Bayes classifier accuracy : ", accuracy*100,'%')
        
        cm = confusion_matrix(y_test, gnb_predictions) 
        
    else:
        print("Wrong number")
        
        
    return cm,accuracy


def ConfAcc(cm):
    
        print("Confusion matrix: ")
        A_BCancer=('Actual Malignant','Actual Benign')
        P_BCancer=('Predict Malignant','Predict Benign')
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(cm)
        ax.grid(False) 
        ax.set_yticklabels(A_BCancer, fontsize=12, color='black')
        ax.set_xticklabels(P_BCancer, fontsize=12, color='black')
        ax.xaxis.set(ticks=range(2))
        ax.yaxis.set(ticks=range(2))
        ax.set_ylim(1.5, -0.5)
         
        for i in range(2):
            
            for j in range(2):
                
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
                
        plt.show()     
        

        

print("This program is performing classification on the basis of a set of data'Breast Cancer Wisconsin'")
print("Select the appropriate classification method by entering the appropriate number: ")
print("1-SVM")
print("2-KNN")
print("3-Naive Bayes classifier")
choice = int(input())

cm_1,accuracy_1=method(choice)

ConfAcc(cm_1)



# In[ ]:





# In[ ]:




