# -*- coding:utf-8 -*-
import csv
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot
import matplotlib.pylab as plt

#create a dataframe 'ts && Convert ts['date'] from string to datetime. You can use ts.index.

#method 1 write data directly

ts = [
    {'date': '2016-05-01 10:23:05.069722', 'tick_numbers': 3213},
    {'date': '2016-05-01 10:23:05.119994', 'tick_numbers': 4324},
    {'date': '2016-05-02 10:23:05.178768', 'tick_numbers': 2132},
    {'date': '2016-05-02 10:23:05.230071', 'tick_numbers': 43242},
    {'date': '2016-05-02 10:23:05.230071', 'tick_numbers': 4234},
    {'date': '2016-05-02 10:23:05.280592', 'tick_numbers': 4324},
    {'date': '2016-05-03 10:23:05.332662', 'tick_numbers': 4324},
    {'date': '2016-05-03 10:23:05.385109', 'tick_numbers': 1245},
    {'date': '2016-05-04 10:23:05.436523', 'tick_numbers': 1555},
    {'date': '2016-05-04 10:23:05.486877', 'tick_numbers': 543345},
]
def time_format(x):
    dt = datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    minute = (dt.minute // 15) * 15
    return datetime(dt.year, dt.month, dt.day, dt.hour, minute, dt.second, dt.microsecond) + timedelta(minutes=15)

ts = pd.DataFrame(ts).fillna(0)
ts['date'] = ts['date'].apply(time_format)
print (ts)

#method 2 write a csv datafile
datafile = file('ts.csv', 'wb')
writer = csv.writer(datafile)
writer.writerow(['date', 'tick_numbers'])
data = [
    ('2016-05-01 10:23:05.069722', 3213),
    ('2016-05-01 10:23:05.119994', 4324),
    ('2016-05-02 10:23:05.178768', 2132),
    ('2016-05-02 10:23:05.230071', 43242),
    ('2016-05-02 10:23:05.230071', 4234),
    ('2016-05-02 10:23:05.280592', 4324),
    ('2016-05-03 10:23:05.332662', 4324),
    ('2016-05-03 10:23:05.385109', 1245),
    ('2016-05-04 10:23:05.436523', 1555),
    ('2016-05-04 10:23:05.486877', 543345)
]
writer.writerows(data)
datafile.close()
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S.%f')
ts = pd.read_csv('ts.csv',index_col='date',date_parser=dateparse)
print(ts.head())
print(ts.head().index)

#Delete useless column with the command del
for col in ts.columns:
    if 'Unnamed' in col:
        del ts[col]
print(ts)

#Print all data from 2016
print (ts['2016'])

#Print all data from May 2016
print (ts['2016-05':])

#Data after May 3rd, 2016
print (ts['2016-05-03':])

#Remove all the data after May 2nd, 2016 using truncate
print(ts.truncate(after='2016-05-02'))

#Count the number of data per timestamp
print(ts.index.value_counts().sort_index())

#Mean value of ticks per day. You will use resample with a period of D and a method of mean.
mean = ts.resample('D').mean()
print (mean)

#Total value ticks per day. You will use sum and a period of D
total = ts.resample('D').sum()
print(total)

#Plot of the total of ticks per day
per = ts.resample('D').sum()
plt.plot(per)
plt.show()
#Create another dataframe
idx = pd.date_range('4/1/2012', '6/1/2012')
df = pd.DataFrame({'ARCA': np.random.randint(low=20000,high=30000,size=62),
                   'BARX': np.random.randint(low=20000,high=30000,size=62)},
                  index=idx)
print(df)

#Truncate the dataframe to get data (before='2012-04-04',after='2012-05-24'),Change the offset of the dataframe by pd.DateOffset(months=1, days=1)
df=df.truncate(before='2012-04-04',after='2012-05-24')
df.index += pd.DateOffset(months=1, days=1)
print(df.head())

#Shift the dataframe by 1 day
print(df.shift(1).head())

#Lag a variable 1 day
print(df['ARCA'].shift(-1).head())

#Aggregate into 2W-SUN (bi-weekly starting by Sunday) by summing up the value of each daily volumw
print(df.resample('2W-SUN').sum())

#Aggregate into weeks by averaging up the value of each daily volume
print(df.resample('2W-SUN').mean())