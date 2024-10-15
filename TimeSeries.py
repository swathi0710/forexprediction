import streamlit as st
st. set_page_config(layout="wide")
col1, col2 = st.columns([1, 3], gap = 'medium')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import math
from datetime import datetime, timedelta
import plotly.offline as po
import main_functions as mfn

@st.cache_data
def gen():
    data = mfn.load_data()
    data["A"]=[str(a).split("/")[0] for a in data["slug"]]
    data["B"]=[str(a).split("/")[1] for a in data["slug"]]
    A_options=list(sorted(data["A"].unique()))
    return A_options, data

with col1:
    st.markdown("""
    <p align="center">
    <img width="230" height="150" src="https://github.com/TelRich/Currency-Foreign-Exchange-Rates/raw/main/image/2023-02-09-09-35-08.png">
    </p>\b\n
    \b\n
    \b\n
    <font size="4"> **Select Currency Pair** </font>\n
    """, unsafe_allow_html=True)
    A_options, data=gen()
    cur_A = st.selectbox('Select first currency', A_options)
    A_data=data[data["A"]==cur_A]
    B_options = list(sorted(A_data["B"].unique()))
    cur_B = st.selectbox('Select second currency', B_options)
    
    Alist=[]
    for b in A_data["B"].unique():
      #print(b,A_data[A_data["B"]==b].shape)
      Alist.append(A_data[A_data["B"]==b])

    cur_map={x:y for y,x in zip(Alist,A_data["B"].unique())}

    A_B=cur_map[cur_B]
    A_B=A_B.set_index(A_B.date)
    A_B.index=pd.to_datetime(A_B.date)

    #upsample to weekly records using mean
    weekly = A_B.resample('W', label='left',closed = 'left').mean(numeric_only=True)
    weekly["close"]=weekly["close"].ffill()
    df_close = weekly['close']

#@st.experimental_memo
def test_stationarity(timeseries,cur_A,cur_B):
    rolmean = timeseries.rolling(52).mean()
    rolstd = timeseries.rolling(52).std()

    chart=pd.DataFrame(timeseries,rolmean,rolstd)
    st.markdown("""
    <font size="4"> **Results of Dickey-Fuller Test** </font>
    \b\n
    """, unsafe_allow_html=True)
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','Number of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    st.write(output)
    stat=["stationary","non-stationary"][int(output["p-value"]>0.05)]
    st.write(f"The ADF test shows that the time series for {cur_A}/{cur_B} close value is {stat}")



#stationarising:
df_log = np.log(df_close)
#Split test and train data
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
#auto-tune using auto arima:
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

p,q,d = model_autoARIMA.order

model_ = ARIMA(train_data, order=(p,q,d))
fitted = model_.fit()
samples=len(test_data)
fc=fitted.forecast(samples, alpha=0.05)
fc2=fitted.forecast(500, alpha=0.05)
fc_series = pd.Series(fc, index=test_data.index)

chart=pd.DataFrame(np.exp(test_data))
chart["Predicted Close values"]=np.exp(fc_series)

mse_arima = round(mean_squared_error(test_data, fc),4)
mae_arima = round(mean_absolute_error(test_data, fc),4)
rmse_arima = round(math.sqrt(mean_squared_error(test_data, fc)),4)
mape1 = round(np.mean(np.abs(fc - test_data)/np.abs(test_data)),4)

train_df=pd.DataFrame(train_data)
train_df["ds"]=train_df.index
train_df["y"]=train_df["close"]

model = Prophet(seasonality_mode='additive', weekly_seasonality=True, daily_seasonality=True )
model.fit(train_df)

future = model.make_future_dataframe(periods=len(test_data), freq='W-SUN',include_history=False)
forecast = model.predict(future)


fs=pd.Series(forecast["yhat"])
fs.index=forecast.ds
chart["Predicted Close values Prophet"]=np.exp(fs)
test_data_ = np.where(test_data == 0, 1e-8, test_data)

mse_prophet = round(mean_squared_error(test_data_, fs),4)
mae_prophet = round(mean_absolute_error(test_data_, fs),4)
rmse_prophet = round(math.sqrt(mean_squared_error(test_data_, fs)),4)
mape = round(np.mean(np.abs(fs - test_data_)/np.abs(test_data_)),4)

data = {
    "PROPHET":[mse_prophet, mae_prophet, rmse_prophet, mape],
    "ARIMA":[mse_arima, mae_arima, rmse_arima, mape1]
}
index = ['MSE', 'MAE', 'RMSE', 'MAPE']
eval=pd.DataFrame(data=data, index=index)

future2=model.make_future_dataframe(periods=500, freq='W-SUN',include_history=False)
forecast2= model.predict(future2)
ts1=pd.Series(train_df["y"])

forecast2= model.predict(future2)
forecast2=forecast2.set_index(forecast2.ds)
       
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

    # visualization
    fig = px.line(weekly, y="close", x=weekly.index,
    title=f"Visualization of {cur_A}/{cur_B} Close Prices Over The Years",
    labels={
    'close':'Close Price',
    'date':'Date'
    })
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The optimal p, q, d values chosen by AUTOARIMA are {p}, {q} and {d}.")
    
    # visualization
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train_data.index, y=np.exp(train_data.values), name="Training Data"))
    fig2.add_trace(go.Scatter(x=test_data.index, y=np.exp(test_data.values), name="Testing Data"))
    fig2.add_trace(go.Scatter(y=chart["Predicted Close values"],x=chart.index, name="ARIMA Forecast"))

    fig2.add_trace(go.Scatter(
        y=chart["Predicted Close values Prophet"],
        x=fs.index,
        name="PROPHET Forecast"
    ))

    fig2.update_layout(
        title={
            'text': "Performance of ARIMA & FB Prophet On The same Test Data",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Close Price",
        font=dict(size=14)
    )

    st.plotly_chart(fig2, use_container_width=True)
    
    if mape1>mape:
         #st.write("The FB Prophet model performs better on an average.")
         ts2=pd.Series(forecast2["yhat"])

    else:
         #st.write("The ARIMA model performs better on an average.")
         ts2=fc2
        
    diff = mape1-mape
    
    if diff < 0:
        st.write(f'The ARIMA model with {round(mape1, 5)} mape performs better than the Prophet model with {round(mape, 5)} mape by {round(abs(diff/mape)*100, 2)}%.')
    else:
        st.write(f'The FBProphet model with {round(mape, 5)} mape performs better than the ARIMA model with {round(mape1, 5)} mape by {round((diff/mape1)*100, 2)}%. ')
        
    ts1=pd.concat([ts1,ts2])
    ts1=np.exp(ts1)    
    
    st.markdown("""
    \b\n
    <font size="5"> **Check for The Predicted Close Price for a Specific Date** </font>\n
    """, unsafe_allow_html=True)
    user_input=st.date_input("Choose a date", value=None)
    #st.write(type(user_input))
    w=weekly.index.to_pydatetime()
    start_date=train_df.iloc[0]["ds"]
    start_date=start_date.date()
    #st.write(type(start_date))
    sample_date = start_date + (((user_input - start_date) // timedelta(weeks=1)) * timedelta(weeks=1))
    sample_date=pd.DatetimeIndex([sample_date]).normalize()
    day_calc=sample_date[0]
    l=ts1[day_calc]
    st.write(f"Predicted Exchange Rate: {l}")
    A = st.number_input(f"Enter the amount in {cur_A}")
    B = A*l
    st.write(f"The estimated value of {A } {cur_A}  is {B} {cur_B} on {user_input.strftime('%B %d, %Y')}")
    
with col1:
    test_stationarity(df_close,cur_A,cur_B)
    st.markdown("""
    <font size="4"> **Error Metrics of Models** </font>\n
    """, unsafe_allow_html=True)
    st.write(eval)
