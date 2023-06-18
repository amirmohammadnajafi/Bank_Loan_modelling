#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
import statsmodels.api as st
from statsmodels.stats import outliers_influence
import random
from sklearn.utils import shuffle
import math
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
from pandas.plotting import scatter_matrix
import pickle


# In[115]:


data1=pd.read_csv(r"C:\Users\HP 250\Desktop\bank_loan_peersonal\Bank_Personal_Loan_Modelling.csv")


# In[116]:


data
pd.set_option('display.max_columns', None)


# In[11]:


data


# In[117]:


data.describe()


# In[118]:


print(data[data["Personal Loan"]==0].count())
print(data[data["Personal Loan"]==1].count())


# In[119]:


data.drop_duplicates(inplace=True)


# In[120]:


data.describe()


# In[121]:


data.corr()


# In[122]:


["CCAvg","CD Account"]


# In[123]:


plt.subplot(2,2,1)
sbn.scatterplot(x=data.Age,y=data.Income,hue=data["Personal Loan"])
plt.subplot(2,2,2)
sbn.scatterplot(x=data.Age,y=data.log_incom,hue=data["Personal Loan"])


# In[124]:


data


# In[34]:


data_Splited=data[data["log_incom"]>=1.7]


# In[50]:


sbn.scatterplot(x=data.Mortgage,y=data["CD Account"],hue=data["Personal Loan"])


# In[41]:


data_Splited["log_Age"]=np.log10(data_Splited.Age)


# In[42]:


data_Splited


# In[125]:


data["log_incom"]=np.log10(data.Income)


# In[126]:


data


# In[127]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.log_incom,data["Mortgage"],data.Age,c=data["Personal Loan"])

plt.show()


# In[108]:


data.to_csv(r"C:\Users\HP 250\Desktop\bank_loan_peersonal\data_loget.csv")


# In[142]:


x=data[["Education","Family","CD Account","Age","log_incom","Mortgage"]]
y=data["Personal Loan"]


# In[143]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[144]:


svm_predic_linear=svm.SVC(kernel='linear')


# In[145]:


svm_predic_linear.fit(x_train,y_train)


# In[146]:


pre=svm_predic_linear.predict(x_test)


# In[147]:


accuracy_score(y_test,pre)


# In[148]:


with open("model_bank_personale_loan_svm1","wb")as file:
    pickle.dump(svm_predic_linear,file)


# In[74]:


cm = confusion_matrix(y_test,pre)

# Create confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot confusion matrix
disp.plot()

# Set plot title and axis labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show plot
plt.show()


# In[85]:


data1
data1["log_incom"]=np.log10(data.Income)
x=data1[["Education","Family","CD Account","Age","log_incom","Mortgage"]]
y=data1["Personal Loan"]


# In[86]:


r=svm_predic_linear.predict(x)


# In[87]:


accuracy_score(r,y)


# In[89]:


cm = confusion_matrix(y,r)

# Create confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot confusion matrix
disp.plot()

# Set plot title and axis labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show plot
plt.show()


# In[ ]:




