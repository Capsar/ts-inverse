# Electricity 321 Hourly
## Source:
[https://zenodo.org/records/4656140 (Electricity 321 Hourly)](https://zenodo.org/records/4656140) at Zenodo.
## Description: 
The electricity dataset represents the electricity consumption of 370 clients recorded in 15-minutes periods in Kilowatt (kW) from 2011 to 2014.
The uploaded dataset is an aggregated version of the original dataset used by Lai et al. (2017). It contains 321 hourly time series from 2012 to 2014.

# Electricity370
## Source:
[https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 (Electricity 370)](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) at UCI.

## Description:
Data set has no missing values.
Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.
Each column represent one client. Some clients were created after 2011. In these cases consumption were considered zero.
All time labels report to Portuguese hour. However all days present 96 measures (24*4). Every year in March time change day (which has only 23 hours) the values between 1:00 am and 2:00 am are zero for all points. Every year in October time change day (which has 25 hours) the values between 1:00 am and 2:00 am aggregate the consumption of two hours.

Data set were saved as txt using csv format, using semi colon (;).
First column present date and time as a string with the following format 'yyyy-mm-dd hh:mm:ss'
Other columns present float values with consumption in kW

## Processing:
1. Replaced all comma's with dots.
2. Replaced all semi-colons with commas.

# KDDCup_2018
## Source:
[https://zenodo.org/records/4656091 (London Smart Meter)](https://zenodo.org/records/4656091) at Zenodo.

## Description:
Kaggle London Smart Meters dataset contains 5560 half hourly time series that represent the energy consumption readings of London households in kilowatt hour (kWh) from November 2011 to February 2014.

The original dataset contains missing values. They have been replaced by carrying forward the corresponding last observations (LOCF method).


# LondonSmartMeter

# Source:
[https://zenodo.org/records/4656091 (London Smart Meter)](https://zenodo.org/records/4656091) at Zenodo.

# Description:
Kaggle London Smart Meters dataset contains 5560 half hourly time series that represent the energy consumption readings of London households in kilowatt hour (kWh) from November 2011 to February 2014.

The original dataset contains missing values. They have been replaced by carrying forward the corresponding last observations (LOCF method).