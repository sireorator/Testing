#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')


# In[19]:


new = "C:\\Users\\Hp\\Desktop\\CI 2nd\\DSC 2\\diabetes.csv"


# In[28]:


filename = pd.read_csv(new)
filename.head()


# In[29]:


filename.info()


# In[30]:


filename.describe()


# In[36]:


filename.hist(bins=10,figsize=(10,10))
plt.show


# In[39]:


missing_values = filename.isnull().sum()
missing_values


# # no missing value

# In[34]:


filename.corr()


# In[37]:


sns.heatmap(filename.corr())


# In[38]:


sns.countplot(x='Outcome',data=filename, palette='hls')


# In[42]:


X = filename.iloc[:,0:7]
y = filename.iloc[:,-1]


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=20)


# In[67]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[68]:


y_pred=LogReg.predict(X_test)


# In[69]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# In[70]:


print(classification_report(y_test, y_pred))


# In[71]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .10, random_state=10)


# In[73]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[74]:


y_pred=LogReg.predict(X_test)


# In[75]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# In[76]:


print(classification_report(y_test, y_pred))


# In[77]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=25)


# In[79]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[82]:


y_pred=LogReg.predict(X_test)


# In[83]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# In[84]:


print(classification_report(y_test, y_pred))


# In[85]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[ ]:




