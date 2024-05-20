#!/usr/bin/env python
# coding: utf-8

# ## Define Question
# 1. What is the air quality index in Dongsi city?
# 2. How do air quality index vary throughout the year?

# ## Data Wrangling
# ### Gathering Data

# In[1]:


#import library
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Load Data
dongsi_df = pd.read_csv("Data/PRSA_Data_Dongsi_20130301-20170228.csv")
dongsi_df.head()


# ### Assesing Data

# In[4]:


air_df = dongsi_df[["PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM"]]
sns.heatmap(air_df.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[5]:


#Greating Data
dongsi_df.info()


# ### Cleaning Data

# In[6]:


#Ceck missing values
dongsi_df.isna().sum()


# In[7]:


#Handling Missing Values
dongsi_df.interpolate(method="linear",limit_direction="forward",inplace=True)
dongsi_df.wd.fillna(value=dongsi_df.wd.mode,inplace=True)


# In[8]:


#Ceck duplicated data
dongsi_df.duplicated().sum()


# In[9]:


dongsi_df.describe()


# #### Cleaned Data
# * interpolate method for handling missing value
# * checked for and found no duplicated data

# ## Exploratory Data Aalytisis
# 1. Create the formula to find AQI;
# 2. Analyze time-series trend in AQI. 

# #### Formula
# * The AQI calculation uses 7 measures: PM2.5, PM10, SO2, NOx, NH3, CO and O3.
# * For PM2.5 and PM10 the average value in last 24-hrs is used with the condition of having at least 16 values.
# * For SO2 and NO2 the value in last 1-hrs.
# * For CO and O3 the maximum value in last 8-hrs is used.
# * Each measure is converted into a Sub-Index based on pre-defined groups.
# * Sometimes measures are not available due to lack of measuring or lack of required data points.
# * Final AQI is the maximum Sub-Index with the condition that at least one of PM2.5 and PM10 should be     available and at least three out of the six should be available.

# In[10]:


#The Formula
frm = mpimg.imread("Data/formula_idx.png")

plt.imshow(frm)


# #### The Breakpoint for the AQI

# In[11]:


#The breakpont for the AQI
breakpoint = mpimg.imread("Data/breakpoint_aqi.png")

plt.imshow(breakpoint)


# In[12]:


#convert CO (mikrogram per meter qubik) to ppm 
def COppm(x):
    return x/1000

dongsi_df["CO(ppm)"] = dongsi_df["CO"].apply(lambda x: COppm(x))


# In[13]:


#convert O3 (mikrogram per meter qubik) to ppm
def O3ppm(x):
    return x/1000

dongsi_df["O3(ppm)"] = dongsi_df["O3"].apply(lambda x: O3ppm(x))


# In[14]:


# find average variable
dongsi_df["PM2.5_24hr_avg"]=dongsi_df.groupby("station")["PM2.5"].rolling(window = 24 ,min_periods = 16).mean().values
dongsi_df["PM10_24hr_avg"]=dongsi_df.groupby("station")["PM10"].rolling(window = 24 ,min_periods = 16).mean().values
dongsi_df["CO_8hr_max"]=dongsi_df.groupby("station")["CO(ppm)"].rolling(window = 8 ,min_periods = 1).max().values
dongsi_df["O3_8hr_max"]=dongsi_df.groupby("station")["O3(ppm)"].rolling(window = 8 ,min_periods = 1).max().values


# #### PM2.5 (Particulate Matter 2.5-microgram per meter qubik)

# In[15]:


def get_PM25_subindex(x):
    if x <= 12:
        return (50/12) * x 
    elif x <= 35.4:
        return (49/23.3) * (x - 12.1) + 51
    elif x <= 55.4:
        return (49/19.9) * (x - 35.5) + 101
    elif x <= 150.4:
        return (49/94.9) * (x - 55.5) + 151
    elif x <= 250.4:
        return  (99/99.9) * (x - 150.5) + 201
    elif x <= 350.4 :
        return (99/99.9) * (x - 250.5) + 301
    elif x > 350.5 :
        return (99/149.9) * (x - 350.5) + 401
    else:
        return 0

dongsi_df["PM2.5_SubIndex"] = dongsi_df["PM2.5_24hr_avg"].apply(lambda x: get_PM25_subindex(x))


# #### PM10 (Particulate Matter 10-microgram per meter qubik)

# In[16]:


def get_PM10_subindex(x):
    if x <= 54:
        return (50/54) * x
    elif x <= 154:
        return (49/99) * (x - 55) + 51
    elif x <= 254:
        return (49/99) * (x - 155) + 101
    elif x <= 354:
        return (49/99) * (x - 255) + 151
    elif x <= 424:
        return (99/69) * (x - 355) + 201
    elif x <= 504:
        return (99/79) * (x - 425) + 301
    elif x > 504:
        return (x - 505) + 401
    else:
        return 0
    
dongsi_df["PM10_SubIndex"]=dongsi_df["PM10_24hr_avg"].apply(lambda x: get_PM10_subindex(x))


# #### SO2 (Sulphur Dioxide-part per billions)

# In[17]:


def get_SO2_subindex(x):
    if x <= 35:
        return (49/35) * x
    elif x <= 75:
        return (49/39) * (x - 36) + 51
    elif x <= 185:
        return (49/109) * (x - 76) + 101
    elif x <= 304:
        return (49/118) * (x - 186) + 151
    elif x <= 604:
        return (99/299) * (x - 305) + 201
    elif x <= 804:
        return (99/199) * (x - 605) + 301
    elif x > 804:
        return (99/199) * (x - 805) + 401
    else:
        return 0

dongsi_df["SO2_SubIndex"] = dongsi_df["SO2"].apply(lambda x: get_SO2_subindex(x))


# #### NO2 (Nitrogen di-oxide - part per billions)

# In[18]:


def get_NO2_subindex(x):
    if x <= 53:
        return x * 50 / 53
    elif x <= 100:
        return 51 + (x - 54) * 49 / 46
    elif x <= 360:
        return 101 + (x - 101) * 49 / 259
    elif x <= 649:
        return 151 + (x - 361) * 49 / 288
    elif x <= 1249:
        return 201 + (x - 650) * 99 / 599
    elif x <= 1649:
        return 301 + (x - 1250) * 99 / 399
    elif x > 1549:
        return 401 + (x - 1650) * 99 / 499
    else:
        return 0

dongsi_df["NO2_SubIndex"] = dongsi_df["NO2"].apply(lambda x: get_NO2_subindex(x))


# #### CO (Carbon Monoxide - part per millions )

# In[19]:


def get_CO_subindex(x):
    if x <= 4.4:
        return x * 50 / 4.4
    elif x <= 9.4:
        return 51 + (x - 4.5) * 49 / 4.9
    elif x <= 12.4:
        return 101 + (x - 9.5) * 49 / 2.9
    elif x <= 15.4:
        return 151 + (x - 12.5) * 49 / 2.9
    elif x <= 30.4:
        return 201 + (x - 15.5) * 99 / 14.9
    elif x <= 40.4:
        return 301 + (x - 30.5) * 99 / 9.9
    elif x > 40.4:
        return 401 + (x - 40.5) * 99 / 9.9
    else:
        return 0

dongsi_df["CO_SubIndex"] = dongsi_df["CO_8hr_max"].apply(lambda x: get_CO_subindex(x))


# #### O3 (Ozone - part per millions)

# In[20]:


def get_O3_subindex(x):
    if x <= 0.054:
        return x * 50 / 0.054
    elif x <= 0.070:
        return 51 + (x - 0.055) * 49 / 0.015
    elif x <= 0.085:
        return 101 + (x - 0.071) * 49 / 0.014
    elif x <= 0.105:
        return 151 + (x - 0.086) * 49 / 0.019
    elif x >105:
        return 201 + (x - 0.106) * 99 / 94
    else:
        return 0

dongsi_df["O3_SubIndex"] = dongsi_df["O3_8hr_max"].apply(lambda x: get_O3_subindex(x))


# In[21]:


#AQI Bucketing
def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "moderate"
    elif x <= 150:
        return "Unhealty for Sensitive Groups"
    elif x <= 200:
        return "Unhealty"
    elif x <= 300:
        return "Very Unhealty"
    elif x > 300:
        return "Hazardous"
    else:
        return np.NaN

dongsi_df["Checks"] = (dongsi_df["PM2.5_SubIndex"] > 0).astype(int) + (dongsi_df["PM10_SubIndex"] > 0).astype(int) + (dongsi_df["O3_SubIndex"] > 0).astype(int) + (dongsi_df["NO2_SubIndex"] > 0).astype(int) + (dongsi_df["CO_SubIndex"] > 0).astype(int) +  (dongsi_df["O3_SubIndex"] > 0).astype(int)

dongsi_df["AQI_calculated"] = round(dongsi_df[["PM2.5_SubIndex", "PM10_SubIndex", "SO2_SubIndex", "NO2_SubIndex", "CO_SubIndex", "O3_SubIndex"]].max(axis = 1))
dongsi_df.loc[dongsi_df["PM2.5_SubIndex"] + dongsi_df["PM10_SubIndex"] <= 0, "AQI_calculated"] = np.NaN
dongsi_df.loc[dongsi_df.Checks < 2, "AQI_calculated"] = np.NaN

dongsi_df["AQI_bucket_calculated"] = dongsi_df["AQI_calculated"].apply(lambda x: get_AQI_bucket(x))
dongsi_df[~dongsi_df.AQI_calculated.isna()].head(13)


# In[22]:


dongsi_df[~dongsi_df.AQI_calculated.isna()].AQI_bucket_calculated.value_counts()


# In[23]:


aqi = mpimg.imread("Data/aqi.jpg")

plt.imshow(aqi)


# In[24]:


dongsi_df["date"] = pd.to_datetime(dongsi_df[["year","month","day","hour"]])
AQI_trends = dongsi_df[["date","AQI_calculated"]].set_index("date").resample("M").mean()

plt.figure(figsize=(15,8))
plt.plot(AQI_trends.index, AQI_trends["AQI_calculated"])
plt.title("Air Quality Index")
plt.ylabel("AQI qalqulated")
plt.xlabel("date")
plt.yticks(ticks = range (100,300,25), labels = [str(x) for x in range (100,300,25)])
plt.grid(True)
plt.show()


# In[ ]:


seasonal_trends = dongsi_df.groupby("month")["AQI_calculated"].mean()

plt.figure(figsize=(15,6))
seasonal_trends.plot(kind="bar", color="skyblue")
plt.title("Average Air Quality Index by Month")
plt.xlabel("Month")
plt.ylabel("Average Air Quality Index")
plt.xticks(ticks=range(0,12), labels = [str(m) for m in range(1,13)], rotation=0)
plt.grid(True)
plt.show()

print(seasonal_trends)


# ## Conclusion
# 1. Air quality index in Dongsi City provides results (index/hour):
# * Unhealthy 17162
# * Unhealty for Sensitive Groups 6086
# * moderate 6041
# * Very Unhealthy 3651
# * Dangerous 1512
# * Good 597
# 2. Trends the air quality index in Dongsi City experience increases and decreases in certain months of each years. High air quality index occurs at the beginning and end of the year, and decreases in the middle of the year.

# In[ ]:




