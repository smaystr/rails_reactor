Homework 5
-----

Examine the following 2 datasets and perform EDA on them:

1. [Heart Disease UCI dataset from HW3](http://ps2.railsreactor.net/datasets/medicine/)
2. [Household electric power consumption UCI dataset](http://ps2.railsreactor.net/datasets/power_consumption/)

<b>Homework should be submitted as 2 separate Jupyter notebooks (.ipynb files, notebook for each dataset) with completed tasks provided below.</b>

For <b>Heart Disease dataset</b>:

1. download the dataset (train and test files)
2. load and concatenate those datasets via pandas
3. print the dataset size and first rows from the dataset
4. print the lists of numeric, categorical and boolean columns
5. analyze values distribution <b>for each numerical column</b>
6. analyze frequency distribution of values <b>for each categorical and boolean columns</b>
7. analyze correlation between columns
8. perform preprocessing if needed (standardization, encodings etc)
9. generate automatic report for the dataset via [pandas_profiling](https://github.com/pandas-profiling/pandas-profiling) library

<b>Household electric power consumption dataset.</b>
This is time series dataset and it's intendeted to be used for regression modeling to predict `Global_active_power` value based on historical data. You should do the following:

1. download the dataset (train and test files)
2. load the dataset via pandas
3. print the dataset size and first rows from the dataset
4. print number of rows with missing values, analyze which columns have the biggest number of missing values
5. analyze values distribution for each column (except `Time` and `Date`)
6. analyze the change in daily average `Global_active_power`. is there any seasonal behaviour? did consumption change with years?
7. for 2 selected dates (2008-02-01 and 2008-02-02) plot the following variables with respect to time:
    * global active power
    * combined plot with measurements of 3 submeters
    * voltage
    * current intensity
    * active energy consumed every minute (in watt hour) not measured in sub-meterings 1, 2 and 3; this can be computed by the given formula:

    ```
    global_active_power * 1000 / 60 - sub_metering_1 - sub_metering_2 - sub_metering_3
    ```

8. detect dates with abnormal voltage or current if any
9. analyze correlation between columns
11. perform preprocessing for the dataset by using any of the methods from lecture 4. the result of this step must be a dataset (X and y) that can be used for modelling
12. generate automatic report for the dataset via [pandas_profiling](https://github.com/pandas-profiling/pandas-profiling) library

<b>Important!</b> Apart from given tasks each notebook should be concluded with your personal judgement about which columns you would use for your model after this analysis and why.

<b>Warning! While doint the assignment, please, keep in mind the following:</b>
1) Be cautious when processing the second dataset - some columns include mixed types (`str` and `float`) so you need to do some data cleaning first
2) <b>DO NOT</b> push datasets or any other items except Jupyter Notebooks with reports to the repo

For visualiation tasks feel free to use any plot type and library that you consider to be appropriate. However, the following libraries are suggested:

* [pandas](https://pandas.pydata.org/) for data manipulations
* [seaborn](https://seaborn.pydata.org/) for plots
* [pandas_profiling](https://github.com/pandas-profiling/pandas-profiling) for automatic report generation

## Information about Power consumption dataset

This dataset contains measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. Different electrical quantities and some sub-metering values are available. 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 (47 months).

#### Attribute Information:
1. Date
2. Time
3. Global_active_power: household global minute-averaged active power (in kilowatt)
4. Global_reactive_power: household global minute-averaged reactive power (in kilowatt)
5. Voltage: minute-averaged voltage (in volt)
6. Global_intensity: household global minute-averaged current intensity (in ampere)
7. Sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
8. Sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
9. Sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.


### Deadline - 11.07.2019
---
### Please ask all questions regarding homework in Discord
