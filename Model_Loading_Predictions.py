#!/usr/bin/env python
# coding: utf-8

# In[4]:


def load_model(X):
    data = X
    import pandas as pd
    import numpy
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # model_uri = "models:/<model_name>/<version>"
    from mlflow import MlflowClient
    client = MlflowClient()
    model_metadata = client.get_latest_versions("MLflow_ka_model", stages=["None"])
    # latest_model_version = model_metadata[0].version
    model_uri = model_metadata[0].source
    model_uri
    model = mlflow.pyfunc.load_model(model_uri)
    
    
    start_index = data.index.get_loc("2015-07-31")
    predictions = []
    
    for sample in range(781, 822):
#     print(X.iloc[sample])
        for i in range(1,6):
            X[f'Sales_lag_{i}'].iloc[sample] = X['Sales'].iloc[sample-i]
        y_fore = pd.Series(model.predict(X.drop(columns = ["Sales"]).iloc[sample].to_numpy().reshape(1, -1)))
        predictions.append(y_fore)
        X['Sales'].iloc[sample] = y_fore
        
    import plotly.graph_objects as go
    trace_actual = go.Scatter(x=X.index, y=X.Sales, mode='lines+markers', name='Actual Values', line=dict(color='blue'))
    fig = go.Figure(data=[trace_actual])
    fig.show()
    
    return X


# In[5]:


# load_model()


# In[ ]:




