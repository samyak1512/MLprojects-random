#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Data_Ingestion import load_data
from Data_Transformation import transform_data
from Model_Loading_Predictions import load_model
from Model_Training import train_model


# In[2]:


train_model()
# this file will run every 2 weeks and trains a new model version


# In[3]:


df = load_data("source")
# this file will input data for the model


# In[5]:


df = transform_data(df)
# this file transforms the data into the format compatible with the model


# In[7]:


y_pred = load_model(df)
# this file acts as a client and loads the model. The loaded model is used to make predictions


# In[11]:


y_pred
# predictions output for the model


# In[ ]:




