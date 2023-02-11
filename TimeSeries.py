import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

url='https://drive.google.com/file/d/18-4rLXTR2B-KVsGTMkngcYQUwhnM0Rod/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
data = pd.read_csv(url)

st.title('Forex rate prediction Dashboard')

data["A"]=[str(a).split("/")[0] for a in data["slug"]]
data["B"]=[str(a).split("/")[1] for a in data["slug"]]

A_options=data["A"].unique()


cur_A = st.selectbox(
     'Select the first currency',
     A_options)
     
A_data=data[data["A"]==cur_A]
     
B_options = A_data["B"].unique()

cur_B = st.selectbox(
     'Select the second currency',
     B_options)
     
Alist=[]
for b in A_data["B"].unique():
  #print(b,A_data[A_data["B"]==b].shape)
  Alist.append(A_data[A_data["B"]==b])

cur_map={x:y for y,x in zip(Alist,A_data["B"].unique())}

A_B=cur_map[cur_B]
A_B=A_B.set_index(A_B.date)
A_B.index=pd.to_datetime(A_B.date)

#upsample to weekly records using mean
weekly = A_B.resample('W', label='left',closed = 'left').mean()

weekly["close"]=weekly["close"].ffill()

st.write(f"Visualization of the {cur_A}/{cur_B} close prices over the years")
st.line_chart(weekly["close"])


df_close = weekly['close']

def test_stationarity(timeseries):
    rolmean = timeseries.rolling(52).mean()
    rolstd = timeseries.rolling(52).std()
    
    datf=pd.DataFrame({"y1":timeseries,"y2":rolmean,"y3":rolstd,"x":timeseries.index})
    
    st.line_chart(datf[[['x', 'y1', 'y2','y3']]
    st.write("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    st.write(output)
test_stationarity(df_close)


