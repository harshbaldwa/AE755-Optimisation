import pandas as pd
import numpy as np
def read_windrose():
    df = pd.read_csv("./code/common/wind_data.csv")
    columns = df.columns[1:]
    speed = np.array(df.columns[1:].astype('float64'))*5/18*2.5
    theta = df['theta']*np.pi/180
    wind_prob = np.array(df[columns])
    wind_prob = wind_prob/np.sum(wind_prob)

    return speed, theta, wind_prob

if __name__=="__main__":
    a, b, c = read_windrose()
    print(a)
    print(b)
    print(c.sum())
