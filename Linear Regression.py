#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib as mp
import statsmodels.api as sm


# In[3]:


mu, sigma = 0, 5 # mean and standard deviation of normal distribution for the error term
x = np.random.uniform(40,80,100)
epsilon = np.random.normal(mu,sigma,100)
y = 3 + 4*x + epsilon


# In[5]:


model_reg = sm.OLS(y,x).fit()
model_reg.summary()


# In[9]:


x_updated = sm.add_constant(x)
model_updated = sm.OLS(y,x_updated).fit()
model_updated.summary()


# In[16]:


# We now generate autocorrelated error terms
epsilon[0] = np.random.normal(mu,sigma,1)
for i in range(0,99):
    epsilon[i+1]=0.4*epsilon[i]+0.6*np.random.normal(mu,sigma,1)


# In[17]:


y = 3 + 4*x + epsilon


# In[18]:


x_updated = sm.add_constant(x)
model_OLS = sm.OLS(y,x_updated).fit()
model_OLS.summary()


# In[26]:


from scipy.linalg import toeplitz
toeplitz(np.array([1,0.5,0,0,0,0,0,0]))


# In[33]:


rho = 0.4
cov_matrix = sigma**2*toeplitz(np.append([1,rho],np.zeros(98)))
sm.GLS(y,x_updated,cov_matrix).fit().summary()


# In[30]:


np.append([1,rho],np.zeros(98))


# In[ ]:




