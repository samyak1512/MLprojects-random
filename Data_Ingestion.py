#!/usr/bin/env python
# coding: utf-8

# In[3]:


def load_data(source):
    import numpy as np
    import pandas as pd
    import logging

    df = pd.read_csv("test.csv", index_col = "Date", low_memory=False, parse_dates=['Date'])
    df_2 = pd.read_csv("train.csv", index_col = "Date", low_memory=False, parse_dates=['Date'])

    df = pd.concat([df, df_2])
    # df_3 = df_2.append(df)
    
    return df

