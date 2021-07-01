#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[3]:


favorite_load = pickle.load(open('./saves/favorite_save.pkl','rb'))
print(favorite_load)

# In[4]:


# 원래 저장했던 성격을 그대로 갖고 있음!
print(type(favorite_load))


# In[5]:


print(favorite_load['tiger'])


# ## lr.pkl file load

# In[8]:


autompg_lr = pickle.load(open('./saves/autompg_lr.pkl', 'rb'))


# In[9]:


print(type(autompg_lr))


# In[10]:


# input from outside
a = 3504.0
b = 8
import numpy as np
pre = np.array([[a,b]])
autompg_lr.predict(pre)

autompg_lr.predict([[3504.0,8]])


# In[ ]:




