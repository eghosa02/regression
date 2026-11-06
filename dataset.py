import numpy as np
import pandas as pd 
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

def load_dataset(ticker):
    data = pd.read_csv(f"{ticker}.csv", index_col=0, parse_dates=True)
    return data