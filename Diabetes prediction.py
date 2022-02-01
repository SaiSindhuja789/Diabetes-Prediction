#!/usr/bin/env python
# coding: utf-8

# In[368]:


#  KNearest Neighbors


# In[1]:


import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
os.chdir("E:\\lang\\ml\\verzeo")


# In[3]:


import pandas as pd


# In[4]:


df=pd.read_csv("diabetes.csv")


# In[5]:


df.head()


# In[8]:


df.shape


# In[9]:


df.corr()


# In[10]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[12]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)


# In[14]:


knn.fit(x_train,y_train)


# In[15]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=knn.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[16]:


accuracy_score(y_test,y_pred)


# In[17]:


#Support Vector Machine


# In[18]:


import os
os.chdir("E:\\lang\\ml\\verzeo")


# In[19]:


import pandas as pd


# In[20]:


df=pd.read_csv("diabetes.csv")


# In[21]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[24]:


from sklearn.svm import SVC


# In[25]:


s=SVC(kernel="rbf",random_state=0)
s.fit(x_train,y_train)


# In[26]:


s1=SVC(kernel="linear",random_state=0)
s1.fit(x_train,y_train)


# In[27]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=s.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

print(cm)


# In[28]:


y_pred1=s1.predict(x_test)
cm1=confusion_matrix(y_test,y_pred1)
cm1


# In[29]:


#using  rbf kernel 
accuracy_score(y_test,y_pred)


# In[30]:


#using linear kernel
accuracy_score(y_test,y_pred1)

