#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:


df = pd.DataFrame(iris.data,columns = iris.feature_names)


# In[4]:


df['target'] = iris.target
df


# In[5]:


x = df.drop('target', axis = 1)
y = df['target']


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)


# In[8]:


y_pred = dtc.predict(x_test)


# In[9]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




