from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import LinearRegression

from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns

sns.set_style(style = 'whitegrid')
sns.set(rc = {'figure.figsize': (12,8), 
             'axes.facecolor': 'white',
             'axes.grid': True,'grid.color': '.7',
             'axes.linewidth': 1.0, 
             'grid.linestyle': u'-'}, font_scale = 1.0)
custom_colors = [ '#616063', '#513e5c', '#a9bcc6','#976f4f', '#616063']  #['#616063','#171717','#2f6761','#a88358',  '#cfc0ef', '#b8e3ea'] #  '#616063', '#513e5c', '#a9bcc6','#976f4f', '#616063', ] #b#d4c9c7 f#cec3eb #d4c9c7
sns.set_palette(custom_colors)



import pandas as pd
import numpy as np

def metrics(real, forecast):
    
    if type(real)==pd.core.frame.DataFrame:
        real=real[real.columns[0]].values
    
    print("Test für Stationarität:")
    dftest = adfuller(real-forecast, autolag='AIC')
    print("\tT-Statistik = {:.3f}".format(dftest[0]))
    print("\tP-Walue = {:.3f}".format(dftest[1]))
    print("Kritische Werte :")
    for k, v in dftest[4].items():
        print("\t{}: {} - Daten sind {} stationär mit einer Wahrscheinlichkeit von  {}% ".format(k, v, "не" if v<dftest[0] else "", 100-int(k[:-1])))
    
    forecast=np.array(forecast)
    print('MAD:', round(abs(real-forecast).mean(),4))
    print('MSE:', round(((real-forecast)**2).mean(),4))
    print('MAPE:', round((abs(real-forecast)/real).mean(),4))
    print('MPE:', round(((real-forecast)/real).mean(),4))
    print('Standartfehler:', round(((real-forecast)**2).mean()**0.5,4))