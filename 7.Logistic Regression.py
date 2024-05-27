#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')


# ### 1. Data Exploration:

# In[2]:


train=pd.read_csv('Titanic_train.csv')
train


# In[3]:


train.head()


# In[4]:


train.tail()


# In[5]:


train.describe().T


# In[6]:


train.hist(figsize=(15,10),bins=30 , edgecolor='black');


# In[7]:


train.boxplot()


# ### 2. Data Preprocessing:

# In[8]:


train.info()


# In[9]:


train.isnull().sum()


# In[10]:


mean=train['Age'].mean()
mean


# In[11]:


train['Age']=train['Age'].fillna(mean)


# In[12]:


train.duplicated().sum()


# In[13]:


df=train[['Survived','Pclass','Sex','Age']]
df=pd.get_dummies(df, columns=['Pclass','Sex']).astype(int)
df


# ### 3. Model Building:

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[15]:


y=df['Survived']
x=df.drop('Survived', axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[16]:


model=LogisticRegression()
model.fit(x_train,y_train)


# In[17]:


model.score(x_test,y_test)


# ### 4. Model Evaluation:

# In[18]:


# the model performed during testing.

from sklearn.metrics import confusion_matrix

y_predicted = model.predict(x_test)
confusion_matrix(y_test, y_predicted)


# In[19]:


from sklearn.model_selection import cross_val_predict

predictions=cross_val_predict(model,x_train,y_train,cv=3)
confusion_matrix(y_train,predictions)


# In[20]:


from sklearn.metrics import classification_report

cl_report=classification_report(y_test, y_predicted)
print(cl_report)

# We can also combine precision and recall into one score, which is called the F-score. 


# In[21]:


y_score=model.predict_proba(x_test)[:,1]
y_score


# In[22]:


from sklearn.metrics import roc_curve,auc

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
roc_auc


# In[23]:


plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()






# In[ ]:





# In[ ]:




