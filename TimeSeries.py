import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from datetime import datetime, timedelta

# Custom functions module
import main_functions as mfn

# Streamlit configuration
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 3], gap='medium')

# Cache data loading and preprocessing
@st.cache
def load_and_preprocess_data():
    data = mfn.load_data()
    data["A"] = [str(a).split("/")[0] for a in data["slug"]]
    data["B"] = [str(a).split("/")[1] for a in data["slug"]]
    A_options = list(sorted(data["A"].unique()))
    return A_options, data

# Load data
A_options, data = load_and_preprocess_data()

# Sidebar
with col1:
    st.markdown("""
    <p align="center">
    <img width="230" height="150" src="https://github.com/TelRich/Currency-Foreign-Exchange-Rates/raw/main/image/2023-02-09-09-35-08.png">
    </p>\b\n
    \b\n
    \b\n
    <font size="4"> **Select Currency Pair** </font>\n
    """, unsafe_allow_html=True)
    
    cur_A = st.selectbox('Select first currency', A_options)
    A_data = data[data["A"] == cur_A]
    B_options = list(sorted(A_data["B"].unique()))
    cur_B = st.selectbox('Select second currency', B_options)

    # Data selection and preprocessing
    A_B = A_data[A_data["B"] == cur_B].set_index("date")
    A_B.index = pd.to_datetime(A_B.index)
    weekly = A_B.resample('W', label='left', closed='left').mean(numeric_only=True)
    weekly["close"] = weekly["close"].ffill()
    df_close = weekly['close']

# Function to test stationarity
def test_stationarity(timeseries, cur_A, cur_B):
    rolmean = timeseries.rolling(52).mean()
    rolstd = timeseries.rolling(52).std()

    st.markdown("""
    <font size="4"> **Results of Dickey-Fuller Test** </font>
    \b\n
    """, unsafe_allow_html=True)

    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'Number of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    st.write(output)

    stat = "stationary" if output["p-value"] <= 0.05 else "non-stationary"
    st.write(f"The ADF test shows that the time series for {cur_A}/{cur_B} close value is {stat}")

# Data transformation and splitting
df_log = np.log(df_close)
train_data, test_data = df_log[3:int(len(df_log) * 0.9)], df_log[int(len(df_log) * 0.9):]

# ARIMA model training and forecasting
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf', max_p=3, max_q=3, m=1, d=None,
                      seasonal=False, start_P=0, D=0, trace=True,
                      error_action='ignore', suppress_warnings=True, stepwise=True)
p, q, d = model_autoARIMA.order

model_arima = ARIMA(train_data, order=(p, q, d)).fit()
fc_arima = model_arima.forecast(len(test_data))
fc_series_arima = pd.Series(fc_arima, index=test_data.index)

# Prophet model training and forecasting
train_df = pd.DataFrame(train_data)
train_df["ds"] = train_df.index
train_df["y"] = train_df["close"]

model_prophet = Prophet(seasonality_mode='additive', weekly_seasonality=True, daily_seasonality=True)
model_prophet.fit(train_df)
future = model_prophet.make_future_dataframe(periods=len(test_data), freq='W-SUN', include_history=False)
forecast_prophet = model_prophet.predict(future)
fc_series_prophet = pd.Series(forecast_prophet["yhat"], index=forecast_prophet.ds)

# Model evaluation
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mae, rmse, mape

mse_arima, mae_arima, rmse_arima, mape_arima = calculate_metrics(test_data, fc_series_arima)
mse_prophet, mae_prophet, rmse_prophet, mape_prophet = calculate_metrics(test_data, fc_series_prophet)

# Main content
with col2:
    st.markdown("""
    <center>
        <font size="7">
            <span style='color:#00a5a7'><b>CURRENCY FX RATE PREDICTION WEB-APP</b></span>
        </font>
    </center>
    \b
    """, unsafe_allow_html=True)
    st.markdown("""
    <font size="4"> A HDSC Fall '22 Capstone Project - Team PyCaret </font>
    * <font size="4"> [**Dataset**](https://www.kaggle.com/datasets/dhruvildave/currency-exchange-rates) </font>
    * <font size="4"> [**GitHub**](https://github.com/TelRich/Currency-Foreign-Exchange-Rates) </font>
    * <font size="4"> [**Medium**](https://medium.com/@rshm.jp07/currency-foreign-exchange-rate-prediction-4630a118de0c) </font>
    ***
    """, unsafe_allow_html=True)

    # Visualization
    fig = px.line(weekly, y="close", x=weekly.index,
                  title=f"Visualization of {cur_A}/{cur_B} Close Prices Over The Years",
                  labels={'close': 'Close Price', 'date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"The optimal p, q, d values chosen by AUTOARIMA are {p}, {q} and {d}.")

    # Combined forecast visualization
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train_data.index, y=np.exp(train_data.values), name="Training Data"))
    fig2.add_trace(go.Scatter(x=test_data.index, y=np.exp(test_data.values), name="Testing Data"))
    fig2.add_trace(go.Scatter(y=np.exp(fc_series_arima), x=test_data.index, name="ARIMA Forecast"))
    fig2.add_trace(go.Scatter(y=np.exp(fc_series_prophet), x=test_data.index, name="PROPHET Forecast"))

    fig2.update_layout(
        title={'text': "Performance of ARIMA & FB Prophet On The same Test Data",
               'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Close Price",
        font=dict(size=14)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Model comparison
    if mape_arima > mape_prophet:
        st.write(f'The FBProphet model with {round(mape_prophet, 5)}% MAPE performs better than the ARIMA model with {round(mape_arima, 5)}% MAPE.')
    else:
        st.write(f'The ARIMA model with {round(mape_arima, 5)}% MAPE performs better than the Prophet model with {round(mape_prophet, 5)}% MAPE.')

    # Future forecasting and prediction
    future2 = model_prophet.make_future_dataframe(periods=500, freq='W-SUN', include_history=False)
    forecast2 = model_prophet.predict(future2)
    ts1 = pd.concat([train_df["y"], forecast2["yhat"]])
    ts1 = np.exp(ts1)

    st.markdown("""
    \b\n
    <font size="5"> **Check for The Predicted Close Price for a Specific Date** </font>\n
    """, unsafe_allow_html=True)

    user_input = st.date_input("Choose a date", value=None)
    start_date = train_df.iloc[0]["ds"].date()
    sample_date = start_date + (((user_input - start_date) // timedelta(weeks=1)) * timedelta(weeks=1))
    l = ts1[sample_date]
    st.write(f"Predicted Exchange Rate: {l}")

    A = st.number_input(f"Enter the amount in {cur_A}")
    B = A * l
    st.write(f"The estimated value of {A} {cur_A}  is {B} {cur_B} on {user_input.strftime('%B %d, %Y')}")

with col1:
    test_stationarity(df_close, cur_A, cur_B)

    st.markdown("""
    <font size="4"> **Error Metrics of Models** </font>\n
    """, unsafe_allow_html=True)

    data = {"PROPHET": [mse_prophet, mae_prophet, rmse_prophet, mape_prophet],
            "ARIMA": [mse_arima, mae_arima, rmse_arima, mape_arima]}
    index = ['MSE', 'MAE', 'RMSE', 'MAPE']
    eval_df = pd.DataFrame(data=data, index=index)
    st.write(eval_df.style.format("{:.4f}"))
