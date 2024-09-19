import pandas as pd

RainyDayPath = 'aDataCollection/RainyDay.csv'
SunnyDayPath = 'aDataCollection/SunnyDay.csv'


RainyDayDF = pd.read_csv(RainyDayPath)
SunnyDayDF = pd.read_csv(SunnyDayPath)

# print(RainyDayDF.isnull().sum())
# print(SunnyDayDF.isnull().sum())

