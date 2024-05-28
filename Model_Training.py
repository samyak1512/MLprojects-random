#!/usr/bin/env python
# coding: utf-8

# In[4]:


def train_model():
    import numpy as np
    import pandas as pd
    import os
    from urllib.parse import urlparse
    import seaborn as sns
    import logging
    import plotly.express as px
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    # get_ipython().run_line_magic('matplotlib', 'inline')
    import mlflow
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    df = pd.read_csv("train.csv", index_col = "Date", low_memory=False, parse_dates=['Date'])
    
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        std_dev = np.std(pred - actual)
        variance = np.var(pred - actual)
        return rmse, mae, r2, std_dev, variance
    
#     data transformation
    store_number_er = 1
    df = df[df["Store"] == store_number_er]
    df = df.asfreq('D')
#     sort the values by date and drop stores when closed
    df.sort_values(by="Date", inplace=True)
    df = df.drop(df[df['Open'] == 0].index)
# converting to date time object
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day

    # Convert the columns to integers
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    
# one hot encoding

    df = pd.get_dummies(df, columns=["StateHoliday"])
    df = pd.get_dummies(df, columns=["DayOfWeek"])
# dropping columns 
    df = df.drop(columns=["Store"])
    # df= df.drop(columns = ["SchoolHoliday"])
    df= df.drop(columns = ["SchoolHoliday", "Promo", "Open", "Customers"])
    df_1 = df.drop(columns = ["Sales"], axis = 1)
# fourier analysis
    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
    fourier = CalendarFourier(freq="A", order = 24)  # 10 sin/cos pairs for "A"nnual seasonality
    dp = DeterministicProcess(
        index=df_1.index,
        constant=False,
        order=0,  # Linear trend
        additional_terms=[fourier],
        drop=True,
    )

    X = dp.in_sample()
#     merging X with df
    X = df.reset_index().merge(X, on='Date').set_index('Date')

    # days within a week
    X["day"] = X.index.dayofweek  # the x-axis (freq)
    X["week"] = X.index.isocalendar().week
    X["week"] = X["week"].astype("int64")  # the seasonal period (period)

    # days within a year
    X["dayofyear"] = X.index.dayofyear
    X["year"] = X.index.year

#     creating specific lags
    def make_specific_lags(df, column, lags):
        lagged_data = pd.concat(
            {
                f'{column}_lag_{i}': df[column].shift(i)
                for i in lags
            },
            axis=1
        )
        return pd.concat([df, lagged_data], axis=1)



    X = make_specific_lags(X,'Sales',  [1,2, 3,4, 5])
    X  = X.fillna(0.0)

    # train-test-split
    y = df.Sales.copy()
    X = X.drop(columns=["Sales"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    
#     set MLflow library

    mlflow.set_experiment("MLflow Quickstart")
    
# training loop
    # Fit and predict
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = pd.Series(model.predict(X_train), index=y_train.index)
        y_fore = pd.Series(model.predict(X_test), index=y_test.index)

        (rmse, mae, r2, std_dev, variance) = eval_metrics(y_test, y_fore)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                model_info = mlflow.sklearn.log_model(model, "MLflowRossmanStoreSalesData", registered_model_name="MLflow_ka_model")
        else:
            model_info = mlflow.sklearn.log_model(model, "MLflowRossmanStoreSalesData")



# In[ ]:




