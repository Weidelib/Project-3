import pandas as pd

weather = pd.read_csv("local_weather.csv", index_col="DATE")

weather

## Lets user see how many data spots are null as a percentage for each catogory
weather.apply(pd.isnull).sum()/weather.shape[0]


## creates new table with core data set as specifyed by NOAA
cw = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
cw.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]


## Shows user how many entrys in the new table are null
core_weather.apply(pd.isnull).sum()

## Shows users the amount of each 
core_weather["snow_depth"].value_counts()

## Fills all null values with the value zero
core_weather["snow"] = core_weather["snow"].fillna(0)

## Fills all null values with the value zero
core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)

## recheck how many null
core_weather.apply(pd.isnull).sum()

## Inspect data table for the input of 9999 as this a placeholder similar to Null this information is specifyed by NOAA
core_weather.apply(lambda x: (x == 9999).sum())

## Inspects which data type is used for each data column
core_weather.dtypes

core_weather.index

##Formats data set to make easier for computer to read
core_weather.index = pd.to_datetime(core_weather.index)

core_weather.index

core_weather.index.year

## Show graph of the max and min temps throughout the timeline
core_weather[["temp_max", "temp_min"]].plot()

## Gives number count of entries per year.
core_weather.index.year.value_counts().sort_index()

core_weather["precip"].plot()

core_weather.groupby(core_weather.index.year).apply(lambda x: x["precip"].sum()).plot()

core_weather["target"] = core_weather.shift(-1)["temp_max"]

core_weather

core_weather = core_weather.iloc[:-1,:].copy()

core_weather

from sklearn.linear_model import Ridge

reg = Ridge(alpha=.1)

predictors = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]

## Sets all data entries dated as before the set date as the training data set
train = core_weather.loc[:"2020-12-31"]

## Sets all data entries dated as after the set date as the test data set
test = core_weather.loc["2021-01-01":]

train

test

reg.fit(train[predictors], train["target"])

predictions = reg.predict(test[predictors])

from sklearn.metrics import mean_squared_error


##Shows users adverage amount off by the prediction made
mean_squared_error(test["target"], predictions)

## Creates the table combined with predictions next to the actual temperature recorded
combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]

##Shows user the predictions next to the actual temperature recorded
combined

##shows the combined table as a graph
combined.plot()

## Show user how each catogory affects the temperature
reg.coef_


