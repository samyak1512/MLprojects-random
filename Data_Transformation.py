#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def transform_data(df):
    import numpy as np
    import pandas as pd
    
    def make_specific_lags(df, column, lags):
        lagged_data = pd.concat(
            {
                f'{column}_lag_{i}': df[column].shift(i)
                for i in lags
            },
            axis=1
        )
        return pd.concat([df, lagged_data], axis=1)

    store_number_er = 1
    df = df[df["Store"] == store_number_er]
    df = df.asfreq('D')
    df.sort_values(by="Date", inplace=True)
    df = df.drop(df[df['Open'] == 0].index)
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day

    # Convert the columns to integers
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df = pd.get_dummies(df, columns=["StateHoliday"])
    df = pd.get_dummies(df, columns=["DayOfWeek"])

    df = df.drop(columns=["Store"])
    df= df.drop(columns = ["SchoolHoliday", "Promo", "Open"])
    
    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

    fourier = CalendarFourier(freq="A", order = 24)  # 10 sin/cos pairs for "A"nnual seasonality

    # dp = DeterministicProcess(
    #     index=df.index,
    #     constant=True,               # dummy feature for bias (y-intercept)
    #     order=1,                     # trend (order 1 means linear)
    #     seasonal=True,               # weekly seasonality (indicators)
    # #     additional_terms=[fourier],  # annual seasonality (fourier)
    #     drop=True,                   # drop terms to avoid collinearity
    # )

    dp = DeterministicProcess(
        index=df.index,
        constant=False,
        order=0,  # Linear trend
    #     seasonal=True,  # Include seasonal dummies
        additional_terms=[fourier],
    #     fourier=2,  # Include Fourier terms
        drop=False,
    )

    X = dp.in_sample()
    # X.info()
    
    X = df.reset_index().merge(X, on='Date').set_index('Date')

    # days within a week
    X["day"] = X.index.dayofweek  # the x-axis (freq)
    X["week"] = X.index.isocalendar().week
    X["week"] = X["week"].astype("int64")  # the seasonal period (period)

    # days within a year
    X["dayofyear"] = X.index.dayofyear
    X["year"] = X.index.year



    X = make_specific_lags(X,'Sales',  [1,2, 3,4, 5])
    X  = X.fillna(0.0)
    X=X.drop(columns = ["Id", "Customers"])
    
    return X

