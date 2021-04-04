import pandas as pd
import numpy as np 
def read_windrose():
    df = pd.read_csv("./opti/common/wind_data.csv")
    columns = df.columns[1:]
    speed = np.array(df.columns[1:].astype('float64'))*5/18*2.5
    theta = df['theta']
    wind_prob = np.array(df[columns])
    wind_prob = wind_prob/np.sum(wind_prob)

    return speed, theta, wind_prob
