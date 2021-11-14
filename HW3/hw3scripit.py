#!/usr/bin/env python
# coding: utf-8

# In[107]:


from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

dataset = scipy.io.loadmat("dataset.mat")
dataset_df = pd.DataFrame(dataset["X"], columns= [1,2,3,4])


dataset_df["target"] = dataset["Y"]
dataset_df

dataset_df_X = dataset_df.drop("target",axis=1)
dataset_df_y = dataset_df["target"]


X_train, X_test, y_train, y_test = train_test_split(dataset_df_X, dataset_df_y, test_size=0.5)



def plot_linear_svm(c):
    
    for c_value in c:
        model_svm = SVC(C=c_value, kernel="linear")
        final_cv_score = np.mean(cross_val_score(model_svm, dataset_df_X,dataset_df_y, cv=2))
        linear_svm_score.append(final_cv_score)
    fig, ax1= plt.subplots(figsize=(5,5))
    ax1.plot(c, linear_svm_score)
    ax1.set(xlabel="Values of C", ylabel="Accuracy", title="Linear SVM")
    print(linear_svm_score)
    plt.show()
    print(f"Based on the graph we see that the optimal value of C for linear svm kernel is: {c[linear_svm_score.index(max(linear_svm_score))]}\n")
    



def plot_poly_svm(c, order):
    poly_svm_score=[]
    for c_value in c:
        if(c_value==1):
            for order_value in order:
                model_svm = SVC(C=c_value, kernel="poly", degree=order_value)
                final_cv_score = np.mean(cross_val_score(model_svm, dataset_df_X,dataset_df_y, cv=2))
                poly_svm_score.append(final_cv_score)
                
    fig, ax1= plt.subplots(figsize=(5,5))
    ax1.plot(order, poly_svm_score)
    ax1.set(xlabel="Order", ylabel="Accuracy", title="Poly SVM(C=1)")
    print(poly_svm_score)
    plt.show()
    print(f"Based on the graph we see that the optimal value of order when C=1 for poly svm kernel is: {order[poly_svm_score.index(max(poly_svm_score))]}\n")
    
    
    poly_svm_score=[]
    for order_value in order:
        if(order_value==2):
            for c_value in c:
                model_svm = SVC(C=c_value, kernel="poly", degree=order_value)
                final_cv_score = np.mean(cross_val_score(model_svm, dataset_df_X,dataset_df_y, cv=2))
                poly_svm_score.append(final_cv_score)
                
    fig, ax1= plt.subplots(figsize=(5,5))
    ax1.plot(c, poly_svm_score)
    ax1.set(xlabel="Values of C", ylabel="Accuracy", title="Poly SVM(order=2)")
    print(poly_svm_score)
    plt.show()
    print(f"Based on the graph we see that the optimal value of C when order=2 for poly svm kernel is: {c[poly_svm_score.index(max(poly_svm_score))]}\n")
    
    
    poly_svm_score=[]
    for order_value in order:
        if(order_value==3):
            for c_value in c:
                model_svm = SVC(C=c_value, kernel="poly", degree=order_value)
                final_cv_score = np.mean(cross_val_score(model_svm, dataset_df_X,dataset_df_y, cv=2))
                poly_svm_score.append(final_cv_score)
                
    fig, ax1= plt.subplots(figsize=(5,5))
    ax1.plot(c, poly_svm_score)
    ax1.set(xlabel="Values of C", ylabel="Accuracy", title="Poly SVM(order=3)")
    print(poly_svm_score)
    plt.show()
    print(f"Based on the graph we see that the optimal value of C when order=3 for poly svm kernel is: {c[poly_svm_score.index(max(poly_svm_score))]}\n")
    
    
def plot_rbf_svm(c, gamma):
    rbf_svm_score=[]
    for c_value in c:
        if(c_value==1):
            for gamma_value in gamma:
                model_svm = SVC(C=c_value, kernel="rbf", gamma=gamma_value)
                final_cv_score = np.mean(cross_val_score(model_svm, dataset_df_X,dataset_df_y, cv=2))
                rbf_svm_score.append(final_cv_score)
        
                
    fig, ax1= plt.subplots(figsize=(5,5))
    ax1.plot(gamma, rbf_svm_score)
    ax1.set(xlabel="C Values", ylabel="Accuracy", title="RBF SVM(C=1)")
    print(rbf_svm_score)
    plt.show()
    print(f"Based on the graph we see that the optimal value of gamma when C=1 for rbf svm kernel is: {gamma[rbf_svm_score.index(max(rbf_svm_score))]}\n")
    
    #now we know optimum gamma is : 10
    
    rbf_svm_score=[]
    for gamma_value in gamma:
        if(gamma_value==10):
            for c_value in c:
                model_svm = SVC(C=c_value, kernel="rbf", gamma=gamma_value)
                final_cv_score = np.mean(cross_val_score(model_svm, dataset_df_X,dataset_df_y, cv=2))
                rbf_svm_score.append(final_cv_score)
        
                
    fig, ax1= plt.subplots(figsize=(5,5))
    ax1.plot(c, rbf_svm_score)
    ax1.set(xlabel="C Values", ylabel="Accuracy", title="RBF SVM(gamma=10)")
    print(rbf_svm_score)
    plt.show()
    print(f"Based on the graph we see that the optimal value of C when gamma=10 for rbf svm kernel is: {c[rbf_svm_score.index(max(rbf_svm_score))]}\n")
    


linear_svm_score=[]   
c=[0.001, 0.01, 0.1, 1, 10, 100]
plot_linear_svm(c)

c=[0.001, 0.01, 0.1, 1, 10, 100]
order=[1,2,3,4,5]
plot_poly_svm(c, order)

c=[0.001, 0.01, 0.1, 1, 10, 100]
gamma = [0.001, 0.01, 0.1, 1, 10, 100]
plot_rbf_svm(c, gamma)


grid={'C':[0.001, 0.01, 0.1, 1, 10, 100]}

def plot_linear_svm_grid(grid):
    
    model_svm = SVC(kernel="linear")
    gs_cv=GridSearchCV(estimator=model_svm,
                      param_grid=grid,
                      cv=2,
                      verbose=2,
                      refit=True)
    gs_cv.fit(X_train,y_train)
    y_preds=gs_cv.predict(X_test)
    print(f"\nThe best value for C is:{gs_cv.best_params_}")
    print(classification_report(y_test, y_preds))

plot_linear_svm_grid(grid)


# In[100]:


dataset_df_X.shape, dataset_df_y.shape


# In[101]:


X_train.shape, y_test.shape


# In[ ]: